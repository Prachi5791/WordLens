

export default function HealthIndicator({
  state,
  health,
}) {
  if (state === "checking") {
    return (
      <div className="health health-checking">
        Checking backend…
      </div>
    );
  }

  if (state === "ready") {
    return (
      <div className="health health-ready">
        Backend ready
      </div>
    );
  }

  return (
    <div className="health health-error">
      Backend unavailable
      {health?.message
        ? ` — ${health.message}`
        : ""}
    </div>
  );
}