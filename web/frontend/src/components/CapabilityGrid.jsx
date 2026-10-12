const CAPABILITIES = [
  ["LD", "Leaf disease", "Apple Mosaic, Black Rot, Alternaria, Healthy"],
  ["PA", "Pest analysis", "Xylotrechus, aphids, leafhoppers, spider mite"],
  ["FH", "Fruit health", "Training capability; current ONNX export omits fruit logits"],
  ["YC", "Yield count", "Apple count estimate from the raw detection head"],
];

export function CapabilityGrid() {
  return (
    <section className="capability-grid" aria-label="Model capabilities">
      {CAPABILITIES.map(([code, title, description]) => (
        <article className="capability-card" key={title}>
          <span className="capability-icon" aria-hidden="true">{code}</span>
          <h2>{title}</h2>
          <p>{description}</p>
        </article>
      ))}
    </section>
  );
}
