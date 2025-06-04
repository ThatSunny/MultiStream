import React from "react";

const ResultDisplay = ({ result }) => {
  if (result.loading) {
    return (
      <div className="result" style={{ display: "block" }}>
        <p>Processing your file... Please wait.</p>
      </div>
    );
  }

  if (result.error) {
    return (
      <div className="result" style={{ display: "block" }}>
        <h2>Error:</h2>
        <p>{result.error}</p>
      </div>
    );
  }

  const fileExtension = result.video_url?.split(".").pop().toLowerCase();

  return (
    <div className="result" style={{ display: "block" }}>
      <h2>Processing Complete!</h2>
      <p>
        Your file has been processed
        successfully.
      </p>
      {fileExtension === "mp4" ? (
        <video controls>
          <source src={result.video_url} type="video/mp4" />
        </video>
      ) : (
        <audio controls>
          <source src={result.video_url} type="audio/mp3" />
        </audio>
      )}
    </div>
  );
};

export default ResultDisplay;
