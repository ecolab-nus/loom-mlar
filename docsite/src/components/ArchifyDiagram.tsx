import type {ReactNode} from 'react';
import useBaseUrl from '@docusaurus/useBaseUrl';

import styles from './ArchifyDiagram.module.css';

type ArchifyDiagramProps = {
  src: string;
  title: string;
  description?: string;
};

/** Embed a delivered Archify standalone viewer without duplicating its source. */
export default function ArchifyDiagram({
  src,
  title,
  description,
}: ArchifyDiagramProps): ReactNode {
  const diagramUrl = useBaseUrl(src);

  return (
    <figure className={styles.figure}>
      <div className={styles.frameShell}>
        <iframe
          className={styles.frame}
          src={diagramUrl}
          title={title}
          loading="lazy"
          allow="fullscreen"
          allowFullScreen
        />
      </div>
      <figcaption className={styles.caption}>
        <span>
          <strong>{title}</strong>
          {description ? ` — ${description}` : null}
        </span>
        <a href={diagramUrl} target="_blank" rel="noreferrer">
          Open standalone
        </a>
      </figcaption>
    </figure>
  );
}
