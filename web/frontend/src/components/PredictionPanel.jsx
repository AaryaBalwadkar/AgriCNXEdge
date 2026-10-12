import { ResultRow } from "./ResultRow.jsx";

export function PredictionPanel({ result }) {
  return (
    <section className="panel results-panel" aria-label="Prediction output">
      <div className="panel-heading">
        <div>
          <p className="eyebrow">Prediction</p>
          <h2>Model output</h2>
        </div>
        {result && <span className="latency">{result.latency_ms} ms</span>}
      </div>

      {!result && <p className="empty-state">Upload an image or choose a sample to view model output.</p>}

      {result && (
        <div className="result-list">
          <ResultRow title="Leaf" value={formatClassResult(result.leaf)} />
          <ResultRow title="Pest" value={formatClassResult(result.pest)} />
          <ResultRow title="Fruit" value={formatFruitResult(result.fruit)} muted={!result.fruit} />
          <ResultRow title="Yield" value={`${result.yield_detection?.apple_count ?? "N/A"} apples`} />
          <ResultRow title="Tensor" value={result.yield_detection?.tensor_shape?.join(" x ") || "N/A"} muted />
        </div>
      )}
    </section>
  );
}

function formatClassResult(data) {
  if (!data) return "N/A";
  return `${data.label} (${Math.round(data.confidence * 100)}%)`;
}

function formatFruitResult(data) {
  if (!data?.label) return "No detected apple boxes for fruit classification";
  return `${data.label} (${data.apple_predictions?.length || 0} apple boxes)`;
}
