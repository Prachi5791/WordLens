export default function SelectionExplainAction({
  selection,
  loading,
  onExplain,
}) {
  if (!selection || selection.trim().length < 2) {
    return null;
  }

  return (
    <div className="selection-action" aria-live="polite">
      <span>
        Selected: “{selection.slice(0, 90)}
        {selection.length > 90 ? "…" : ""}”
      </span>

      <button
        type="button"
        onClick={onExplain}
        disabled={loading}
      >
        {loading ? "Explaining…" : "Explain selection"}
      </button>
    </div>
  );
}