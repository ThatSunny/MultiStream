import React, { useState } from "react";
import UploadForm from "./components/UploadForm";
import ResultDisplay from "./components/ResultDisplay";
import "./index.css";

const App = () => {
  const [result, setResult] = useState(null);

  return (
    <div className="container">
      <header>
        <h1>Multi-Lingual Stream</h1>
        <p>A Speech-to-Speech Streaming Converter in 12 Languages</p>
      </header>

      <UploadForm setResult={setResult} />
      {result && <ResultDisplay result={result} />}
    </div>
  );
};

export default App;
