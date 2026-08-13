import type {ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';

import styles from './index.module.css';

const features = [
  {
    title: 'Model architectures',
    description:
      'Compose memories, compute processors, data movers, resources, and scale-out networks.',
  },
  {
    title: 'Connect compiler IR',
    description:
      'Parse processor functionality from MLIR and export complete architectures as adl.* modules.',
  },
  {
    title: 'Evaluate and visualize',
    description:
      'Evaluate symbolic schedules and export architecture graphs for the interactive web viewer.',
  },
];

function FeatureCards(): ReactNode {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {features.map((feature) => (
            <div className={clsx('col col--4', styles.feature)} key={feature.title}>
              <div className={styles.featureCard}>
                <Heading as="h2">{feature.title}</Heading>
                <p>{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home(): ReactNode {
  return (
    <Layout
      title="Hardware architecture modeling"
      description="Documentation for MLAR, the Multi-Level Architecture Representation">
      <header className={styles.hero}>
        <div className="container">
          <p className={styles.eyebrow}>Multi-Level Architecture Representation</p>
          <Heading as="h1" className={styles.heroTitle}>
            Describe hardware for compiler flows.
          </Heading>
          <p className={styles.heroSubtitle}>
            A Rust library for structured hardware models, MLIR integration,
            symbolic performance evaluation, and architecture visualization.
          </p>
          <div className={styles.actions}>
            <Link className="button button--primary button--lg" to="/docs/architecture-concepts">
              Explore the concepts
            </Link>
            <Link className="button button--secondary button--lg" to="/docs/installation">
              Install MLAR
            </Link>
          </div>
        </div>
      </header>
      <main>
        <FeatureCards />
      </main>
    </Layout>
  );
}

