import React, { useState } from "react";

const UploadForm = ({ setResult }) => {
  const [file, setFile] = useState(null);
  const [targetLanguage, setTargetLanguage] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!file || !targetLanguage) {
      setResult({ error: "Please select a file and target language." });
      return;
    }

    const formData = new FormData();
    formData.append("file", file);
    formData.append("target_language", targetLanguage);

    setResult({ loading: true });

    try {
      const response = await fetch("http://localhost:5000/", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        setResult({ video_url: data.video_url });  // Changed from videoUrl to video_url
      } else {
        setResult({ error: data.error || "Failed to process file." });
      }
    } catch (error) {
      console.error("Upload error:", error);
      setResult({ error: "Failed to process. Try again." });
    }
  };

  return (
    <div className="upload-form">
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Upload your file (MP4 or MP3):</label>
          <input
            type="file"
            accept=".mp4,.mp3"
            onChange={(e) => setFile(e.target.files[0])}
            required
          />
        </div>

        <div className="form-group">
          <label>Enter Target Language (e.g. Hindi, French, Spanish):</label>
          <input
            type="text"
            value={targetLanguage}
            onChange={(e) => setTargetLanguage(e.target.value)}
            required
          />
        </div>

        <button type="submit" className="submit-btn">
          Upload and Process
        </button>
      </form>
    </div>
  );
};

export default UploadForm;