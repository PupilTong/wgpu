//! Awaiting JavaScript promises, and running Rust futures on the JavaScript event
//! loop.
//!
//! `wasm-bindgen-futures` builds its executor on `queueMicrotask` reached through a
//! wasm-bindgen import. The mechanism here is the same — a promise's `then`
//! callbacks complete the future, and woken tasks are drained from a microtask —
//! but every JavaScript call goes through Node-API, and the callbacks are real JS
//! functions created by [`crate::closure`].
//!
//! Everything is single-threaded on purpose: a `napi_env` belongs to one thread,
//! so a future spawned here is polled on the thread that spawned it, by the
//! JavaScript event loop that thread is running.
//!
//! # Relationship to napi-rs's own async support
//!
//! napi-rs has `Promise<T>`, `PromiseRaw`, `#[napi] async fn` and `AsyncTask`, and
//! [`JsFuture`] below is built the same way `napi::Promise<T>` is: `then`/`catch`
//! callbacks on the same thread, feeding a one-shot slot that the future polls
//! (see `napi/src/bindgen_runtime/js_values/promise.rs`). Two things keep this
//! crate from using it directly:
//!
//! * `napi::Promise<T>` resolves to a `T: FromNapiValue`, and the values here are
//!   [`crate::value::JsValue`] handles converted by [`crate::convert::FromJs`].
//!   Bridging the two traits would mean depending on the whole `napi` crate to
//!   gain nothing this file does not already do in a dozen lines.
//! * Nothing in napi-rs polls a future on the JavaScript event loop.
//!   `#[napi] async fn` and `AsyncTask` hand work to napi-rs's Tokio runtime or the
//!   libuv pool, which is right for an addon's own boundary but wrong for
//!   [`spawn_local`], whose contract is "run on this thread, between JavaScript
//!   jobs" — `wgpu`'s `Queue::on_submitted_work_done` relies on that.
//!
//! An addon built on this crate should still use napi-rs's async support for its
//! own exported functions; the two coexist, since each drives its own futures.

use alloc::boxed::Box;
use alloc::collections::VecDeque;
use alloc::rc::Rc;
use core::cell::{Cell, RefCell};
use core::future::Future;
use core::pin::Pin;
use core::ptr;
use core::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

use napi_sys as sys;

use crate::closure::{stateless_function, Closure};
use crate::convert::FromJs;
use crate::env;
use crate::js_sys::Promise;
use crate::value::JsValue;

/// A Rust future backed by a JavaScript promise.
///
/// Resolves to `Ok` with the promise's value, or `Err` with the rejection reason.
pub struct JsFuture<T = JsValue> {
    state: Rc<RefCell<Settlement<T>>>,
}

struct Settlement<T> {
    result: Option<Result<T, JsValue>>,
    waker: Option<Waker>,
    /// The `then` callbacks, kept alive until one of them fires. Dropping them
    /// early would be harmless — JavaScript owns its functions here — but holding
    /// them makes the lifetime obvious and matches wasm-bindgen's structure.
    callbacks: Option<SettleCallbacks>,
}

/// The pair of `then` callbacks a [`JsFuture`] installs: one per outcome.
type SettleCallbacks = (Closure<dyn FnMut(JsValue)>, Closure<dyn FnMut(JsValue)>);

impl<T> core::fmt::Debug for JsFuture<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("JsFuture").finish_non_exhaustive()
    }
}

impl<T: FromJs + 'static> JsFuture<T> {
    /// Attaches to any thenable.
    fn attach(thenable: &JsValue) -> Self {
        let state = Rc::new(RefCell::new(Settlement::<T> {
            result: None,
            waker: None,
            callbacks: None,
        }));

        let resolved = {
            let state = Rc::clone(&state);
            Closure::wrap(Box::new(move |value: JsValue| {
                settle(&state, Ok(T::from_js(value)));
            }) as Box<dyn FnMut(JsValue)>)
        };
        let rejected = {
            let state = Rc::clone(&state);
            Closure::wrap(Box::new(move |value: JsValue| {
                settle(&state, Err(value));
            }) as Box<dyn FnMut(JsValue)>)
        };

        // `then` is called before the callbacks are stored, which is safe because
        // JavaScript cannot run a callback until this thread yields.
        let attached = crate::rt::call_method(
            thenable,
            c"then",
            &[
                resolved.as_js_value().clone(),
                rejected.as_js_value().clone(),
            ],
        );
        crate::rt::unwrap_js(attached, "Promise.then");

        state.borrow_mut().callbacks = Some((resolved, rejected));
        Self { state }
    }
}

/// Records a settled value and wakes the task waiting on it.
fn settle<T>(state: &Rc<RefCell<Settlement<T>>>, result: Result<T, JsValue>) {
    let waker = {
        let mut state = state.borrow_mut();
        // The other callback will never fire now, so let both of them go.
        state.callbacks = None;
        if state.result.is_some() {
            log::error!("wgpu-napi-web: a promise settled twice");
            return;
        }
        state.result = Some(result);
        state.waker.take()
    };
    if let Some(waker) = waker {
        waker.wake();
    }
}

impl<T: FromJs + 'static> From<Promise<T>> for JsFuture<T> {
    fn from(promise: Promise<T>) -> Self {
        Self::attach(AsRef::<JsValue>::as_ref(&promise))
    }
}

impl<T> Future for JsFuture<T> {
    type Output = Result<T, JsValue>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.state.borrow_mut();
        match state.result.take() {
            Some(result) => Poll::Ready(result),
            None => {
                state.waker = Some(context.waker().clone());
                Poll::Pending
            }
        }
    }
}

/// Runs `future` to completion on this thread's JavaScript event loop.
///
/// The future is not polled by this call: it is queued and first polled from a
/// microtask, which is what `wasm_bindgen_futures::spawn_local` does and what
/// callers such as `Queue::on_submitted_work_done` rely on (the callback must not
/// run before the caller returns).
pub fn spawn_local<F: Future<Output = ()> + 'static>(future: F) {
    Task::spawn(Box::pin(future));
}

struct Task {
    future: RefCell<Pin<Box<dyn Future<Output = ()>>>>,
    /// Set while the task sits in the queue, so a task woken twice is only queued
    /// once.
    queued: Cell<bool>,
}

thread_local! {
    static QUEUE: RefCell<VecDeque<Rc<Task>>> = const { RefCell::new(VecDeque::new()) };
    /// The JS function that drains [`QUEUE`], created once per thread and kept
    /// alive for as long as the thread runs tasks.
    static DRAIN: RefCell<Option<JsValue>> = const { RefCell::new(None) };
    /// Whether a drain microtask is already scheduled.
    static DRAIN_SCHEDULED: Cell<bool> = const { Cell::new(false) };
}

impl Task {
    fn spawn(future: Pin<Box<dyn Future<Output = ()>>>) {
        let task = Rc::new(Self {
            future: RefCell::new(future),
            queued: Cell::new(false),
        });
        task.enqueue();
    }

    fn enqueue(self: Rc<Self>) {
        if self.queued.replace(true) {
            return;
        }
        QUEUE.with(|queue| queue.borrow_mut().push_back(self));
        schedule_drain();
    }

    /// Polls this task once, dropping it when it completes.
    fn run(self: &Rc<Self>) {
        self.queued.set(false);
        let waker = waker_for(Rc::clone(self));
        let mut context = Context::from_waker(&waker);
        // `borrow_mut` cannot conflict: a task is only ever polled from the drain
        // loop, one at a time, and a waker only re-queues it.
        let mut future = self.future.borrow_mut();
        let _ = future.as_mut().poll(&mut context);
    }
}

/// Polls every queued task. Tasks woken during the drain are queued again and
/// picked up by the next microtask, so one task cannot starve the event loop.
fn drain_queue() {
    DRAIN_SCHEDULED.with(|scheduled| scheduled.set(false));
    let ready: VecDeque<Rc<Task>> = QUEUE.with(|queue| core::mem::take(&mut *queue.borrow_mut()));
    for task in ready {
        task.run();
    }
}

/// Asks JavaScript to drain the queue after the current job finishes.
fn schedule_drain() {
    if DRAIN_SCHEDULED.with(|scheduled| scheduled.replace(true)) {
        return;
    }
    if !env::is_installed() {
        // Nothing can be scheduled without an environment. Leave the flag set so a
        // later `spawn_local` does not spin on this path; the task stays queued and
        // runs at the first drain after `install`.
        log::error!(
            "wgpu-napi-web: spawn_local before wgpu_napi_web::install(env) — \
             the task will not run until an environment is installed"
        );
        DRAIN_SCHEDULED.with(|scheduled| scheduled.set(false));
        return;
    }

    let drain = DRAIN.with(|slot| {
        slot.borrow_mut()
            .get_or_insert_with(|| stateless_function(Some(drain_trampoline)))
            .clone()
    });

    // `queueMicrotask` is in every browser that has WebGPU and in Node ≥ 11;
    // `Promise.resolve().then` is the fallback for a host that lacks it.
    let queue_microtask = crate::rt::global(c"queueMicrotask").unwrap_or(JsValue::UNDEFINED);
    let scheduled = if queue_microtask.is_function() {
        crate::rt::call(&queue_microtask, &JsValue::UNDEFINED, &[drain])
    } else {
        crate::rt::global(c"Promise")
            .and_then(|promise| crate::rt::call_method(&promise, c"resolve", &[]))
            .and_then(|resolved| crate::rt::call_method(&resolved, c"then", &[drain]))
    };
    if let Err(error) = scheduled {
        DRAIN_SCHEDULED.with(|flag| flag.set(false));
        log::error!("wgpu-napi-web: could not schedule the task queue: {error}");
    }
}

/// The JavaScript entry point for [`drain_queue`].
///
/// # Safety
///
/// Called by Node-API as a callback; reads no data pointer and returns no value.
unsafe extern "C" fn drain_trampoline(
    _env: sys::napi_env,
    _info: sys::napi_callback_info,
) -> sys::napi_value {
    drain_queue();
    ptr::null_mut()
}

fn waker_for(task: Rc<Task>) -> Waker {
    // SAFETY: the vtable functions below are implemented for exactly this data
    // pointer type (`Rc<Task>` erased to `*const ()`), and each one restores and
    // consumes or clones the `Rc` correctly.
    unsafe { Waker::from_raw(raw_waker(task)) }
}

fn raw_waker(task: Rc<Task>) -> RawWaker {
    RawWaker::new(Rc::into_raw(task).cast::<()>(), &VTABLE)
}

static VTABLE: RawWakerVTable =
    RawWakerVTable::new(clone_waker, wake_waker, wake_by_ref, drop_waker);

/// # Safety
///
/// `data` must be an `Rc<Task>` pointer created by [`raw_waker`].
unsafe fn clone_waker(data: *const ()) -> RawWaker {
    let task = Rc::from_raw(data.cast::<Task>());
    let clone = Rc::clone(&task);
    // Keep the original refcount owned by the waker being cloned from.
    let _ = Rc::into_raw(task);
    raw_waker(clone)
}

/// # Safety
///
/// `data` must be an `Rc<Task>` pointer created by [`raw_waker`]; this consumes it.
unsafe fn wake_waker(data: *const ()) {
    Rc::from_raw(data.cast::<Task>()).enqueue();
}

/// # Safety
///
/// `data` must be an `Rc<Task>` pointer created by [`raw_waker`]; this borrows it.
unsafe fn wake_by_ref(data: *const ()) {
    let task = Rc::from_raw(data.cast::<Task>());
    Rc::clone(&task).enqueue();
    let _ = Rc::into_raw(task);
}

/// # Safety
///
/// `data` must be an `Rc<Task>` pointer created by [`raw_waker`]; this consumes it.
unsafe fn drop_waker(data: *const ()) {
    drop(Rc::from_raw(data.cast::<Task>()));
}
