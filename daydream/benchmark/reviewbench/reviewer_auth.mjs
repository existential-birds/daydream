/** Bind native Pi 1.3's credential store without relocating its refresh lock.
 * Installed beside the published package's unbundled CLI; both import the same
 * AuthStorage class. Configuration and sessions stay in the private role home.
 */
import { realpathSync } from 'node:fs';
import { isAbsolute, join } from 'node:path';
import { AuthStorage } from './dist/core/auth-storage.js';

const path = process.env.DAYDREAM_RB_PI_AUTH_PATH;
if (!path || !isAbsolute(path) || realpathSync(path) !== path) {
    throw new Error('ReviewBench requires a canonical native Pi credential path');
}
const create = AuthStorage.create.bind(AuthStorage);
const home = process.env.PI_CODING_AGENT_DIR;
if (!home || !isAbsolute(home) || realpathSync(home) !== home) {
    throw new Error('ReviewBench requires a private native Pi configuration directory');
}
const privatePath = join(home, 'auth.json');
AuthStorage.create = (requested) => {
    if (requested !== undefined && requested !== path && requested !== privatePath) {
        throw new Error('ReviewBench reviewer cannot select another credential store');
    }
    return create(path);
};
