export function ResultRow({ muted = false, title, value }) {
  return (
    <div className={muted ? "result-row result-muted" : "result-row"}>
      <strong>{title}</strong>
      <span>{value}</span>
    </div>
  );
}
