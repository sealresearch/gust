// Copies what the package ships from the rest of the repository, so there is one copy of each in git: the licence.
// (The runtime and the registry come with @sealresearch/gust, which this package depends on.)
import { copyFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const pkg = join(dirname(fileURLToPath(import.meta.url)), '..');
copyFileSync(join(pkg, '..', '..', 'LICENSE'), join(pkg, 'LICENSE'));
console.log('LICENSE -> packages/relay/LICENSE');
