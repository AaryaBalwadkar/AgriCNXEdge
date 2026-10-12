export function UploadPanel({ error, file, loading, onFileChange, onSampleSelect, onSubmit, preview, samples }) {
  return (
    <section className="panel upload-panel" aria-label="Image upload">
      <label className="dropzone">
        <span className="upload-icon" aria-hidden="true">UP</span>
        <span className="dropzone-title">{file ? file.name : "Choose crop image"}</span>
        <span className="dropzone-subtitle">JPEG, PNG, or WEBP up to 8 MB</span>
        <input type="file" accept="image/png,image/jpeg,image/webp" onChange={onFileChange} />
      </label>

      <div className="sample-grid" aria-label="Sample images">
        {samples.map((name) => (
          <button className="sample-button" key={name} type="button" onClick={() => onSampleSelect(name)}>
            {name.replace(".jpg", "")}
          </button>
        ))}
      </div>

      {preview && <img className="preview-image" src={preview} alt="Selected crop" />}

      <button className="primary-button" type="button" onClick={onSubmit} disabled={!file || loading}>
        {loading ? "Running inference..." : "Run model"}
      </button>

      {error && <p className="error-message">{error}</p>}
    </section>
  );
}
