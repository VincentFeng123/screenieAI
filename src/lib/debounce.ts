/// Trailing-edge debounce. Repeated calls within `wait` ms reset the timer;
/// only the most recent call's args fire. Used for keychain writes so a
/// pasted-then-typed key doesn't trigger one keyring write per keystroke.
export function debounce<Args extends unknown[]>(
  fn: (...args: Args) => void,
  wait: number,
): ((...args: Args) => void) & { flush: () => void; cancel: () => void } {
  let timer: ReturnType<typeof setTimeout> | null = null;
  let pending: Args | null = null;

  const debounced = (...args: Args) => {
    pending = args;
    if (timer) clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      const a = pending;
      pending = null;
      if (a) fn(...a);
    }, wait);
  };

  debounced.flush = () => {
    if (timer) {
      clearTimeout(timer);
      timer = null;
    }
    const a = pending;
    pending = null;
    if (a) fn(...a);
  };

  debounced.cancel = () => {
    if (timer) clearTimeout(timer);
    timer = null;
    pending = null;
  };

  return debounced;
}
