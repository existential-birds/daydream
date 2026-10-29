/** Bind the pinned evaluator's default factory to Pi's canonical auth path.
 *
 * Pi 0.75.5 supports AuthStorage.create(path), but ReviewBench rc10 calls it
 * without arguments. Keep upstream files untouched and all storage/refresh/lock
 * behavior in Pi. PI_CODING_AGENT_DIR still names fresh disposable configuration.
 * Load this from the judge package so both it and the evaluator use the same SDK.
 */
import { realpathSync } from 'node:fs';
import { isAbsolute } from 'node:path';
import { AuthStorage, FileAuthStorageBackend } from '@earendil-works/pi-coding-agent';

const path = process.env.DAYDREAM_RB_PI_AUTH_PATH;
if (!path || !isAbsolute(path) || realpathSync(path) !== path) {
    throw new Error('ReviewBench requires a canonical Pi credential path');
}
AuthStorage.create = (requested) => {
    if (requested !== undefined && requested !== path) {
        throw new Error('ReviewBench judge cannot select another credential store');
    }
    const backend = new FileAuthStorageBackend(path);
    return AuthStorage.fromStorage({
        withLock(fn) {
            // Pinned Pi's synchronous startup read gives up after ~180ms. A
            // sibling refresh may legitimately hold its lock much longer. Retry
            // only ELOCKED using the same native backend/sidecar, before model
            // discovery can mistake contention for missing subscription auth.
            const deadline = Date.now() + 60_000;
            for (;;) {
                try {
                    return backend.withLock(fn);
                } catch (error) {
                    if (error?.code !== 'ELOCKED' || Date.now() >= deadline) throw error;
                    Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 100);
                }
            }
        },
        withLockAsync: (fn) => backend.withLockAsync(fn),
    });
};
