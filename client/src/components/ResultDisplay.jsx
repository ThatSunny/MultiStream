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

  // Check if we have a video URL or audio URL
  if (!result.video_url && !result.audio_url) {
    return (
      <div className="result" style={{ display: "block" }}>
        <p>No media URL provided</p>
      </div>
    );
  }

  // Create the full URL
  const mediaUrl = result.video_url 
    ? `http://localhost:5000${result.video_url}`
    : `http://localhost:5000${result.audio_url}`;
  
  console.log("Attempting to load media from:", mediaUrl);

  return (
    <div className="result" style={{ display: "block" }}>
      <h2>Processing Complete!</h2>
      <p>Your file has been processed successfully.</p>
      
      {/* Display original and translated text if available */}
      {result.original_text && (
        <div style={{ marginBottom: "20px", padding: "10px", backgroundColor: "#f5f5f5", borderRadius: "5px" }}>
          <h3>Original Text:</h3>
          <p>{result.original_text}</p>
        </div>
      )}
      
      {result.translated_text && (
        <div style={{ marginBottom: "20px", padding: "10px", backgroundColor: "#e8f5e8", borderRadius: "5px" }}>
          <h3>Translated Text:</h3>
          <p>{result.translated_text}</p>
        </div>
      )}

      {/* Display video or audio based on what's available */}
      {result.video_url ? (
        <video 
          controls 
          width="100%" 
          style={{ maxWidth: "800px" }}
          onError={(e) => {
            console.error("Video loading error:", e);
            console.error("Failed URL:", mediaUrl);
          }}
          onLoadStart={() => console.log("Video loading started")}
          onCanPlay={() => console.log("Video can play")}
        >
          <source src={mediaUrl} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      ) : (
        <audio 
          controls 
          style={{ width: "100%", maxWidth: "800px" }}
          onError={(e) => {
            console.error("Audio loading error:", e);
            console.error("Failed URL:", mediaUrl);
          }}
          onLoadStart={() => console.log("Audio loading started")}
          onCanPlay={() => console.log("Audio can play")}
        >
          <source src={mediaUrl} type="audio/mp3" />
          Your browser does not support the audio tag.
        </audio>
      )}
      
      {/* Download link */}
      <div style={{ marginTop: "20px" }}>
        <a 
          href={mediaUrl} 
          download 
          style={{ 
            display: "inline-block",
            padding: "10px 20px",
            backgroundColor: "#007bff",
            color: "white",
            textDecoration: "none",
            borderRadius: "5px"
          }}
        >
          Download {result.video_url ? 'Video' : 'Audio'}
        </a>
      </div>
    </div>
  );
};

export default ResultDisplay;