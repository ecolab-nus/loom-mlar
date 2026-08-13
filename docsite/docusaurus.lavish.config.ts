import type {Config} from '@docusaurus/types';

import baseConfig from './docusaurus.config';

/**
 * Review build for Lavish.
 *
 * The hash router emits a single HTML entry point and relative asset URLs, so
 * the generated site remains functional when Lavish serves the artifact from
 * its session URL. The regular configuration remains the deployment source.
 */
const config: Config = {
  ...baseConfig,
  baseUrl: '/',
  future: {
    ...baseConfig.future,
    experimental_router: 'hash',
  },
};

export default config;
