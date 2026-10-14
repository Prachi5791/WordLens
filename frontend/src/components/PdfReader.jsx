

import {
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  Document,
  Page,
  pdfjs,
} from "react-pdf";

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString();

/*
 * Only reliable lexical Day-2 detections are made
 * interactive in the PDF text layer.
 */
const INTERACTIVE_METHODS = new Set([
  "exact_canonical",
  "alias",
  "normalized_variant",
]);

function normalizeForMatching(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[‐-‒–—−]/g, "-")
    .replace(/[-\s]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function getInteractiveTerms(detectedTerms) {
  return detectedTerms.filter((detection) => {
    return (
      INTERACTIVE_METHODS.has(detection?.extraction_method) &&
      Boolean(detection?.term)
    );
  });
}

function getDetectionVariants(detection) {
  const variants = new Set();

  if (detection?.matched_text) {
    variants.add(
      normalizeForMatching(detection.matched_text),
    );
  }

  if (detection?.term) {
    variants.add(
      normalizeForMatching(detection.term),
    );
  }

  if (Array.isArray(detection?.aliases_matched)) {
    detection.aliases_matched.forEach((alias) => {
      variants.add(normalizeForMatching(alias));
    });
  }

  return [...variants].filter(Boolean);
}

function findExactDetection(text, detections) {
  const normalized = normalizeForMatching(text);

  if (!normalized) {
    return null;
  }

  for (const detection of detections) {
    if (getDetectionVariants(detection).includes(normalized)) {
      return detection;
    }
  }

  return null;
}

function getPageTextSpans(element) {
  const page = element.closest(".react-pdf__Page");

  if (!page) {
    return [];
  }

  return Array.from(
    page.querySelectorAll(
      ".react-pdf__Page__textContent span",
    ),
  );
}

/*
 * PDF.js may split one technical phrase across
 * multiple text-layer spans.
 */
function findDetectionAroundSpan(span, detections) {
  const spans = getPageTextSpans(span);

  if (!spans.length) {
    return null;
  }

  const index = spans.indexOf(span);

  if (index === -1) {
    return null;
  }

  const direct = findExactDetection(
    span.textContent,
    detections,
  );

  if (direct) {
    return {
      detection: direct,
      spans: [span],
    };
  }

  const maxLength = 5;

  for (let length = 2; length <= maxLength; length += 1) {
    const startMin = Math.max(
      0,
      index - length + 1,
    );

    const startMax = Math.min(
      index,
      spans.length - length,
    );

    for (
      let start = startMin;
      start <= startMax;
      start += 1
    ) {
      const group = spans.slice(
        start,
        start + length,
      );

      if (!group.includes(span)) {
        continue;
      }

      const combined = group
        .map((item) => item.textContent || "")
        .join(" ");

      const detection = findExactDetection(
        combined,
        detections,
      );

      if (detection) {
        return {
          detection,
          spans: group,
        };
      }
    }
  }

  return null;
}

export default function PdfReader({
  file,
  detectedTerms = [],
  onSelectionChange,
  onSelectionClear,
  onTermHover,
  onTermLeave,
}) {
  const [numPages, setNumPages] = useState(null);
  const [pageWidth, setPageWidth] = useState(760);
  const [error, setError] = useState("");

  const readerRef = useRef(null);

  const interactiveTerms = useMemo(
    () => getInteractiveTerms(detectedTerms),
    [detectedTerms],
  );

  const fileUrl = useMemo(() => {
    if (!file) {
      return null;
    }

    return URL.createObjectURL(file);
  }, [file]);

  useEffect(() => {
    return () => {
      if (fileUrl) {
        URL.revokeObjectURL(fileUrl);
      }
    };
  }, [fileUrl]);

  useEffect(() => {
    function updateWidth() {
      const availableWidth = window.innerWidth - 80;
      const width = Math.min(820, availableWidth);

      setPageWidth(Math.max(500, width));
    }

    updateWidth();
    window.addEventListener("resize", updateWidth);

    return () => {
      window.removeEventListener("resize", updateWidth);
    };
  }, []);

  /*
   * Hover detection/highlighting.
   */
  useEffect(() => {
    const container = readerRef.current;

    if (!container) {
      return undefined;
    }

    let activeSpans = [];

    function clearHighlight() {
      activeSpans.forEach((span) => {
        span.classList.remove("wordlens-term");
      });

      activeSpans = [];
    }

    function handleMouseOver(event) {
      const target = event.target;

      if (!(target instanceof HTMLElement)) {
        return;
      }

      const span = target.closest(
        ".react-pdf__Page__textContent span",
      );

      if (!span || !container.contains(span)) {
        return;
      }

      const match = findDetectionAroundSpan(
        span,
        interactiveTerms,
      );

      if (!match) {
        return;
      }

      clearHighlight();

      match.spans.forEach((item) => {
        item.classList.add("wordlens-term");
      });

      activeSpans = match.spans;

      const rects = match.spans.map((item) =>
        item.getBoundingClientRect(),
      );

      const rect = {
        left: Math.min(...rects.map((item) => item.left)),
        top: Math.min(...rects.map((item) => item.top)),
        right: Math.max(...rects.map((item) => item.right)),
        bottom: Math.max(...rects.map((item) => item.bottom)),
      };

      onTermHover?.({
        term: match.detection,
        element: span,
        rect,
      });
    }

    function handleMouseMove(event) {
      if (activeSpans.length === 0) {
        return;
      }

      const target = event.target;

      if (target instanceof Node) {
        const stillInsideActiveTerm = activeSpans.some(
          (activeSpan) =>
            activeSpan === target || activeSpan.contains(target),
        );

        if (stillInsideActiveTerm) {
          return;
        }
      }

      clearHighlight();
      onTermLeave?.();
    }

    function handleMouseOut(event) {
      const target = event.target;

      if (!(target instanceof HTMLElement)) {
        return;
      }

      const span = target.closest(
        ".react-pdf__Page__textContent span",
      );

      if (!span || !container.contains(span)) {
        return;
      }

      const related = event.relatedTarget;

      if (
        related instanceof Node &&
        span.contains(related)
      ) {
        return;
      }

      /*
       * When moving between spans of the same PDF text
       * layer, do not hide the explanation immediately.
       */
      const movingWithinActiveTerm =
        related instanceof Node &&
        activeSpans.some(
          (activeSpan) =>
            activeSpan === related ||
            activeSpan.contains(related),
        );

      if (movingWithinActiveTerm) {
        return;
      }

      clearHighlight();
      onTermLeave?.();
    }

    container.addEventListener(
      "mouseover",
      handleMouseOver,
    );

    container.addEventListener(
      "mouseout",
      handleMouseOut,
    );

    container.addEventListener(
      "mousemove",
      handleMouseMove,
    );

    /*
     * A PDF text span can keep the hover state alive because the
     * browser does not always produce the expected span-level
     * mouseout sequence when the pointer leaves the PDF area.
     *
     * Selection is handled separately by App.jsx, so leaving the
     * reader should always clear a hover-only explanation.
     */
    function handleReaderMouseLeave() {
      clearHighlight();
      onTermLeave?.();
    }

    container.addEventListener(
      "mouseleave",
      handleReaderMouseLeave,
    );

    return () => {
      container.removeEventListener(
        "mouseover",
        handleMouseOver,
      );

      container.removeEventListener(
        "mouseout",
        handleMouseOut,
      );

      container.removeEventListener(
        "mousemove",
        handleMouseMove,
      );

      container.removeEventListener(
        "mouseleave",
        handleReaderMouseLeave,
      );

      clearHighlight();
    };
  }, [
    interactiveTerms,
    onTermHover,
    onTermLeave,
  ]);

  /*
   * Keep the explanation card tied to the browser's real text selection.
   * When the user clicks elsewhere and the native selection disappears,
   * notify App.jsx so a selection-owned explanation is removed too.
   */
  useEffect(() => {
    function handleDocumentSelectionChange() {
      const selection = window.getSelection();

      if (!selection || selection.rangeCount === 0) {
        onSelectionClear?.();
        return;
      }

      const text = selection.toString().trim();

      if (!text) {
        onSelectionClear?.();
      }
    }

    document.addEventListener(
      "selectionchange",
      handleDocumentSelectionChange,
    );

    return () => {
      document.removeEventListener(
        "selectionchange",
        handleDocumentSelectionChange,
      );
    };
  }, [onSelectionClear]);

  /*
   * Browser text selection.
   *
   * IMPORTANT:
   * The selection rectangle is calculated HERE.
   * App.jsx therefore never references an undefined
   * `rect`.
   */
  function handleSelection() {
    const selection = window.getSelection();

    if (!selection || selection.rangeCount === 0) {
      return;
    }

    const text = selection
      .toString()
      .replace(/\s+/g, " ")
      .trim();

    if (!text || text.length < 2) {
      return;
    }

    const range = selection.getRangeAt(0);

    if (!readerRef.current?.contains(range.commonAncestorContainer)) {
      return;
    }

    const browserRect = range.getBoundingClientRect();

    const rect =
      browserRect.width === 0 &&
      browserRect.height === 0
        ? null
        : {
            left: browserRect.left,
            top: browserRect.top,
            right: browserRect.right,
            bottom: browserRect.bottom,
          };

    // Capture the actual PDF sentence around the selection. The selected text
    // alone is not enough for context-aware RAG (for example, "Kafka" must be
    // interpreted from the sentence that mentions Apache Kafka/streaming).
    const page = range.commonAncestorContainer.parentElement?.closest(
      ".react-pdf__Page",
    );
    const pageText = page?.innerText?.replace(/\s+/g, " ").trim() || "";
    const sentenceCandidates = pageText.match(/[^.!?]+[.!?]+/g) || [pageText];
    const normalizedSelection = text.toLowerCase();
    const context =
      sentenceCandidates.find((sentence) =>
        sentence.toLowerCase().includes(normalizedSelection),
      )?.trim() || pageText || text;

    onSelectionChange?.({
      text,
      context,
      rect,
    });
  }

  if (!fileUrl) {
    return (
      <div className="reader-empty">
        <div>
          <h2>Research paper reader</h2>
          <p>
            Upload a research-paper PDF to start exploring
            technical terminology.
          </p>
        </div>
      </div>
    );
  }

  return (
    <section className="reader-shell">
      <div className="reader-toolbar">
        <span>
          {numPages
            ? `${numPages} page${numPages === 1 ? "" : "s"}`
            : "Loading PDF…"}
        </span>

        <span className="reader-selection-hint">
          Hover highlighted terms or select text for explanations
        </span>
      </div>

      <div
        ref={readerRef}
        className="pdf-scroll-area"
        onMouseUp={handleSelection}
        onKeyUp={handleSelection}
      >
        <Document
          file={fileUrl}
          suspense={false}
          onLoadSuccess={({ numPages: pages }) => {
            setNumPages(pages);
            setError("");
          }}
          onLoadError={(loadError) => {
            setError(
              loadError?.message ||
                "Unable to render this PDF.",
            );
          }}
          loading={
            <div className="reader-state">
              Loading research paper…
            </div>
          }
          error={
            <div className="reader-state reader-error">
              {error || "Unable to render this PDF."}
            </div>
          }
          noData={
            <div className="reader-state reader-error">
              Invalid or empty PDF.
            </div>
          }
        >
          {Array.from(
            new Array(numPages || 0),
            (_, index) => (
              <div
                className="pdf-page-wrapper"
                key={`page-${index + 1}`}
              >
                <div className="page-number">
                  Page {index + 1}
                </div>

                <Page
                  pageNumber={index + 1}
                  width={pageWidth}
                  renderTextLayer
                  renderAnnotationLayer
                  loading={
                    <div className="reader-state">
                      Loading page {index + 1}…
                    </div>
                  }
                />
              </div>
            ),
          )}
        </Document>
      </div>
    </section>
  );
}

