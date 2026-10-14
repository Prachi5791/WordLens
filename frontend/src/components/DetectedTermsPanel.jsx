export default function DetectedTermsPanel({
  terms,
  selectedTerm,
  onSelect,
}) {
  const uniqueTerms = Array.from(
    new Map(
      terms.map((item) => [
        item.term_id || item.term,
        item,
      ]),
    ).values(),
  );

  return (
    <section className="panel">
      <div className="panel-heading">
        <div>
          <h2>Detected terms</h2>
          <p>
            {uniqueTerms.length} unique technical term
            {uniqueTerms.length === 1 ? "" : "s"}
          </p>
        </div>
      </div>

      {uniqueTerms.length === 0 ? (
        <div className="panel-empty">
          No technical terms were detected in this document.
        </div>
      ) : (
        <div className="term-list">
          {uniqueTerms.map((item) => {
            const isSelected =
              selectedTerm?.term_id === item.term_id;

            return (
              <button
                type="button"
                key={item.term_id || item.term}
                className={`term-item ${
                  isSelected ? "selected" : ""
                }`}
                onClick={() => onSelect(item)}
              >
                <span className="term-name">
                  {item.term}
                </span>

                <span className="term-meta">
                  {item.subdomain || item.domain || "Technical term"}
                </span>

                {item.confidence != null && (
                  <span className="term-confidence">
                    {Math.round(item.confidence * 100)}%
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </section>
  );
}