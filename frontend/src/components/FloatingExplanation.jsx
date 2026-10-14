
import {
  useEffect,
  useMemo,
  useState,
} from "react";

const CARD_WIDTH = 380;
const CARD_HEIGHT = 360;
const GAP = 14;
const VIEWPORT_MARGIN = 12;

function decodeHtmlEntities(value) {
  if (value == null) return "";
  const text = String(value);
  if (!text.includes("&")) return text;
  const textarea = document.createElement("textarea");
  textarea.innerHTML = text;
  return textarea.value;
}

export default function FloatingExplanation({
  state,
  term,
  explanation,
  loading,
  error,
  position,
}) {
  const [cardPosition, setCardPosition] = useState({
    left: 20,
    top: 80,
  });

  const calculatedPosition = useMemo(() => {
    const sourceX = Number.isFinite(position?.x)
      ? position.x
      : 20;

    const sourceY = Number.isFinite(position?.y)
      ? position.y
      : 80;

    const viewportWidth = window.innerWidth;
    const viewportHeight = window.innerHeight;

    /*
     * Prefer the right side of the actual
     * hovered/selected text.
     */
    let left = sourceX;
    let top = sourceY;

    /*
     * If there is not enough room on the right,
     * move the card to the left of the text.
     */
    if (
      left + CARD_WIDTH + VIEWPORT_MARGIN >
      viewportWidth
    ) {
      left = sourceX - CARD_WIDTH - GAP;
    }

    /*
     * Keep the card inside the viewport horizontally.
     */
    left = Math.max(
      VIEWPORT_MARGIN,
      Math.min(
        left,
        viewportWidth -
          CARD_WIDTH -
          VIEWPORT_MARGIN,
      ),
    );

    /*
     * Keep the card inside the viewport vertically.
     * This is what prevents it from being pushed below
     * the screen when the selected word is near the bottom.
     */
    top = Math.max(
      VIEWPORT_MARGIN,
      Math.min(
        top,
        viewportHeight -
          CARD_HEIGHT -
          VIEWPORT_MARGIN,
      ),
    );

    return { left, top };
  }, [position?.x, position?.y]);

  useEffect(() => {
    function updatePosition() {
      setCardPosition(calculatedPosition);
    }

    updatePosition();

    window.addEventListener(
      "resize",
      updatePosition,
    );

    window.addEventListener(
      "scroll",
      updatePosition,
      true,
    );

    return () => {
      window.removeEventListener(
        "resize",
        updatePosition,
      );

      window.removeEventListener(
        "scroll",
        updatePosition,
        true,
      );
    };
  }, [calculatedPosition]);

  if (state !== "visible") {
    return null;
  }

  const displayedTerm =
    explanation?.resolved_term ||
    explanation?.term ||
    term?.term ||
    "Selected text";

  const definition =
    explanation?.canonical_definition ||
    explanation?.definition ||
    null;

  const simpleDefinition =
    explanation?.simplified_definition ||
    null;

  const contextualExplanation =
    explanation?.generated_explanation ||
    null;

  const sources = Array.isArray(
    explanation?.sources,
  )
    ? explanation.sources
    : [];

  const firstSource = sources[0];

  const isContextual =
    explanation?.grounding_status === "grounded";

  return (
    <div
      className="floating-explanation"
      style={{
        position: "fixed",
        left: `${cardPosition.left}px`,
        top: `${cardPosition.top}px`,
        width: `${CARD_WIDTH}px`,
        maxWidth: `calc(100vw - ${
          VIEWPORT_MARGIN * 2
        }px)`,
        maxHeight: `calc(100vh - ${
          VIEWPORT_MARGIN * 2
        }px)`,
        overflowY: "auto",
        zIndex: 99999,
      }}
      role="dialog"
      aria-label="WordLens explanation"
    >
      <div className="floating-header">
        <div className="floating-title-area">
          <span className="floating-label">
            WORDLENS
          </span>

          <h3>{displayedTerm}</h3>
        </div>

        {isContextual && (
          <span className="floating-status grounded">
            CONTEXTUAL
          </span>
        )}
      </div>

      {loading && (
        <div
          className="floating-loading"
          aria-live="polite"
        >
          <span className="loading-dot" />
          Getting meaning…
        </div>
      )}

      {!loading && error && (
        <div
          className="floating-error"
          role="alert"
        >
          {error}
        </div>
      )}

      {!loading && !error && definition && (
        <div className="floating-section">
          <span className="floating-section-title">
            Meaning
          </span>

          <p>{decodeHtmlEntities(definition)}</p>
        </div>
      )}

      {!loading &&
        !error &&
        simpleDefinition && (
          <div className="floating-section">
            <span className="floating-section-title">
              Simple meaning
            </span>

            <p>{decodeHtmlEntities(simpleDefinition)}</p>
          </div>
        )}

      {!loading &&
        !error &&
        contextualExplanation && (
          <div className="floating-section contextual-section">
            <span className="floating-section-title">
              In this context
            </span>

            <p>{decodeHtmlEntities(contextualExplanation)}</p>
          </div>
        )}

      {!loading &&
        !error &&
        !definition &&
        !simpleDefinition &&
        !contextualExplanation && (
          <div className="floating-warning">
            No grounded explanation is available for
            the selected text.
          </div>
        )}

      {!loading &&
        !error &&
        firstSource && (
          <div className="floating-source">
            Source:{" "}
            {firstSource.source_url ? (
              <a
                href={firstSource.source_url}
                target="_blank"
                rel="noreferrer"
              >
                {firstSource.source || "Source"}
              </a>
            ) : (
              firstSource.source || "Source"
            )}
          </div>
        )}
    </div>
  );
}
