

export default function ExplanationCard({
  explanation,
  loading,
  error,
}) {
  if (loading) {
    return (
      <section className="panel explanation-panel">
        <div className="loading-block">
          Retrieving terminology and preparing explanation…
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="panel explanation-panel error-panel">
        <h2>Explanation unavailable</h2>
        <p>{error}</p>
      </section>
    );
  }

  if (!explanation) {
    return (
      <section className="panel explanation-panel">
        <div className="panel-empty">
          Select a detected term or select text in the PDF.
        </div>
      </section>
    );
  }

  const grounded =
    explanation.grounding_status === "grounded";

  return (
    <section
      className="panel explanation-panel"
      tabIndex="-1"
    >
      <div className="explanation-header">
        <div>
          <span className="eyebrow">Explanation</span>
          <h2>
            {explanation.resolved_term ||
              explanation.requested_term ||
              "Selected term"}
          </h2>
        </div>

        <span
          className={`grounding-badge ${
            grounded ? "grounded" : "not-grounded"
          }`}
        >
          {explanation.grounding_status || "unknown"}
        </span>
      </div>

      {explanation.canonical_definition && (
        <div className="definition-block">
          <h3>Definition</h3>
          <p>{decodeHtmlEntities(explanation.canonical_definition)}</p>
        </div>
      )}

      {explanation.simplified_definition && (
        <div className="definition-block">
          <h3>Simple meaning</h3>
          <p>{decodeHtmlEntities(explanation.simplified_definition)}</p>
        </div>
      )}

      {explanation.generated_explanation && (
        <div className="definition-block">
          <h3>Contextual explanation</h3>
          <p>{decodeHtmlEntities(explanation.generated_explanation)}</p>
        </div>
      )}

      {!grounded && (
        <div className="notice">
          The system did not have sufficient retrieval evidence to
          produce a grounded explanation.
        </div>
      )}

      {explanation.sources?.length > 0 && (
        <div className="sources">
          <h3>Sources</h3>

          <ul>
            {explanation.sources.map((source, index) => (
              <li key={`${source.source_url || source.source}-${index}`}>
                {source.source_url ? (
                  <a
                    href={source.source_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {source.source || source.source_url}
                  </a>
                ) : (
                  <span>{source.source || "Source"}</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {explanation.warnings?.length > 0 && (
        <div className="warning-list">
          {explanation.warnings.map((warning, index) => (
            <p key={index}>{warning}</p>
          ))}
        </div>
      )}
    </section>
  );

function decodeHtmlEntities(value) {
  if (value == null) return "";
  const text = String(value);
  if (!text.includes("&")) return text;
  const textarea = document.createElement("textarea");
  textarea.innerHTML = text;
  return textarea.value;
}
}