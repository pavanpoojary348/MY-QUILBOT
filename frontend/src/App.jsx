import { useState } from "react";

function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleParaphrase() {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch("http://127.0.0.1:8000/paraphrase", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div style={{ maxWidth: 600, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>AI Writing Assistant</h1>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type a sentence to paraphrase..."
        rows={4}
        style={{ width: "100%", padding: 8, fontSize: 16 }}
      />

      <button
        onClick={handleParaphrase}
        disabled={loading || !text.trim()}
        style={{ marginTop: 12, padding: "8px 16px", fontSize: 16 }}
      >
        {loading ? "Paraphrasing..." : "Paraphrase"}
      </button>

      {error && (
        <p style={{ color: "red", marginTop: 16 }}>Error: {error}</p>
      )}

      {result && (
        <div style={{ marginTop: 24 }}>
          <p><strong>Original:</strong> {result.original}</p>
          <p><strong>Paraphrased:</strong> {result.paraphrased}</p>
          <p><strong>Confident:</strong> {result.confident ? "Yes" : "No"}</p>
        </div>
      )}
    </div>
  );
}

export default App;