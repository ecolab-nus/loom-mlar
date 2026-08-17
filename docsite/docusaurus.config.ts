import type {Config} from '@docusaurus/types';
import type {Options, ThemeConfig} from '@docusaurus/preset-classic';
import {themes as prismThemes} from 'prism-react-renderer';

const config: Config = {
  title: 'MLAR',
  tagline: 'Multi-Level Architecture Representation for hardware compiler flows',
  favicon: 'img/favicon.svg',

  url: 'https://ecolab-nus.github.io',
  baseUrl: '/loom-mlar/',
  organizationName: 'ecolab-nus',
  projectName: 'loom-mlar',

  onBrokenLinks: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          path: '../docs',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.ts',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Options,
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'MLAR',
      logo: {
        alt: 'MLAR logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'docsSidebar',
          position: 'left',
          label: 'Documentation',
        },
        {
          href: 'https://github.com/ecolab-nus/loom-mlar',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentation',
          items: [
            {label: 'Concepts', to: '/docs/architecture-concepts'},
            {label: 'Installation', to: '/docs/installation'},
            {label: 'Usage', to: '/docs/usage'},
          ],
        },
        {
          title: 'Project',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/ecolab-nus/loom-mlar',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} MLAR contributors.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['rust', 'toml'],
    },
  } satisfies ThemeConfig,
};

export default config;
