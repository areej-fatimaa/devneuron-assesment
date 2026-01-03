import { useState } from "react";
import styles from "./index.module.css";

type AttackResult = {
  clean_prediction: number;
  adversarial_prediction: number;
  attack_success: boolean;
  adversarial_image: string;
};

const DIGIT_LABELS = ["0","1","2","3","4","5","6","7","8","9"];

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [epsilon, setEpsilon] = useState<number>(0.1);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AttackResult | null>(null);
  const [error, setError] = useState<string>("");

  const runAttack = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("epsilon", epsilon.toString());

    try {
      const res = await fetch("http://127.0.0.1:8000/attack", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        throw new Error("Attack request failed");
      }

      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError("Failed to run FGSM attack. Please ensure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className={styles.container}>
      <h1 className={styles.title}>FGSM Adversarial Attack Demo</h1>

      <p className={styles.subtitle}>
        Demonstration of the Fast Gradient Sign Method (FGSM) showing how
        small, targeted perturbations can mislead a neural network.
      </p>

      {/* Upload Section */}
      <div className={styles.card}>
        <label className={styles.label}>Upload Image</label>
        <input
          type="file"
          accept="image/png, image/jpeg"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
      </div>

      {/* Epsilon Section */}
      <div className={styles.card}>
        <label className={styles.label}>
          Epsilon: <strong>{epsilon}</strong>
        </label>
        <input
          type="range"
          min="0"
          max="0.3"
          step="0.01"
          value={epsilon}
          onChange={(e) => setEpsilon(Number(e.target.value))}
        />
        {epsilon > 0.2 && (
          <p className={styles.warning}>
            High epsilon values may introduce visible perturbations.
          </p>
        )}
      </div>

      {/* Run Button */}
      <button
        className={styles.button}
        disabled={!file || loading}
        style={{ opacity: !file || loading ? 0.6 : 1 }}
        onClick={runAttack}
      >
        {loading ? "Running FGSM Attack..." : "Run Attack"}
      </button>

      {error && <p className={styles.error}>{error}</p>}

      {/* Results */}
      {result && (
        <div className={styles.resultCard}>
          <h2>Attack Result</h2>

          <p>
            <strong>Clean Prediction:</strong>{" "}
            {DIGIT_LABELS[result.clean_prediction]}
          </p>

          <p>
            <strong>Adversarial Prediction:</strong>{" "}
            {DIGIT_LABELS[result.adversarial_prediction]}
          </p>

          <p>
            <strong>Attack Success:</strong>{" "}
            {result.attack_success ? "✅ Yes" : "❌ No"}
          </p>

          <div className={styles.images}>
            <div>
              <p>Original Image</p>
              {file && (
                <img
                  src={URL.createObjectURL(file)}
                  alt="Original"
                  width={150}
                />
              )}
            </div>

            <div>
              <p>Adversarial Image</p>
              <img
                src={`data:image/png;base64,${result.adversarial_image}`}
                alt="Adversarial"
                width={150}
              />
            </div>
          </div>

          <p className={styles.explanation}>
            FGSM generates adversarial examples by adding a small perturbation
            in the direction of the gradient of the loss with respect to the
            input image, scaled by epsilon.
          </p>
        </div>
      )}
    </main>
  );
}
