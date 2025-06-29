import React, { useState } from "react";
import {
  Container,
  Box,
  Typography,
  Button,
  CircularProgress,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import CloudUploadIcon from "@mui/icons-material/CloudUpload";
import InfoIcon from "@mui/icons-material/Info";

const UploadButton = styled(Button)(({ theme }) => ({
  backgroundColor: "#66bb6a",
  color: "white",
  fontWeight: "bold",
  fontSize: "1rem",
  padding: theme.spacing(1.5),
  borderRadius: theme.spacing(1),
  boxShadow: "0 4px 12px rgba(102, 187, 106, 0.5)",
  transition: "all 0.3s ease",
  "&:hover": {
    backgroundColor: "#388e3c",
    transform: "scale(1.01)",
  },
}));

function PlantDiseaseDetector() {
  const [image, setImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showInstructions, setShowInstructions] = useState(false);

  const handleImageUpload = (e) => {
    setResult(null);
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreviewUrl(URL.createObjectURL(file));
    }
  };

  const handleSubmit = () => {
    if (!image) {
      alert("Please upload an image of a plant leaf.");
      return;
    }

    setLoading(true);
    const formData = new FormData();
    formData.append("image", image);

    fetch("http://127.0.0.1:5000/detect", {
      method: "POST",
      body: formData,
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Failed to detect disease");
        }
        return response.json();
      })
      .then((data) => {
        setResult(data);
      })
      .catch((error) => {
        console.error("Error:", error);
        setResult({ error: "Error detecting disease. Please try again." });
      })
      .finally(() => {
        setLoading(false);
      });
  };

  return (
    <div
      className="plant-bg"
      style={{
        backgroundImage: "url(/path/to/your/bg-image.jpg)",
        backgroundSize: "cover",
        backgroundPosition: "center",
        height: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <Container
        maxWidth="sm"
        className="glass-box"
        sx={{
          py: 5,
          px: 4,
          backgroundColor: "rgba(255,255,255,0.85)",
          borderRadius: 4,
          boxShadow: "0 8px 24px rgba(0,0,0,0.2)",
          textAlign: "center",
        }}
      >
        <Typography
          variant="h3"
          gutterBottom
          align="center"
          sx={{ fontWeight: 600, color: "green" }}
        >
          🌿 Plant Disease Detector
        </Typography>

        {/* Upload Button */}
        <Box mb={3}>
          <UploadButton
            component="label"
            variant="contained"
            startIcon={<CloudUploadIcon />}
            fullWidth
          >
            Upload Leaf Image
            <input
              type="file"
              accept="image/*"
              hidden
              onChange={handleImageUpload}
            />
          </UploadButton>
        </Box>

        {/* Image Preview */}
        {previewUrl && (
          <Box mb={3}>
            <Typography
              variant="subtitle1"
              sx={{ mb: 1, color: "#2e7d32", fontWeight: 500 }}
            >
              Preview:
            </Typography>
            <img
              src={previewUrl}
              alt="Preview"
              style={{
                width: "100%",
                maxWidth: "300px",
                borderRadius: "12px",
                boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
              }}
            />
          </Box>
        )}

        {/* Detect Button */}
        <Box mb={3}>
          <UploadButton onClick={handleSubmit} fullWidth disabled={loading}>
            {loading ? (
              <CircularProgress size={24} color="inherit" />
            ) : (
              "Detect Disease"
            )}
          </UploadButton>
        </Box>

        {/* Results - Moved here */}
        {result && (
          <Box mb={3}>
            {result.error ? (
              <Typography variant="h6" color="error">
                ❌ {result.error}
              </Typography>
            ) : (
              <>
                <Typography
                  variant="h6"
                  sx={{ fontWeight: 500, color: "green" }}
                >
                  🪴 Plant: {result.plant}
                </Typography>
                <Typography
                  variant="h6"
                  sx={{ fontWeight: 500, color: "green" }}
                >
                  🦠 Disease: {result.disease}
                </Typography>
                <Typography
                  variant="h6"
                  sx={{ fontWeight: 500, color: "green" }}
                >
                  🔬 Confidence: {result.confidence}
                </Typography>
              </>
            )}
          </Box>
        )}

        {/* Instructions Button */}
        <Box mb={3}>
          <UploadButton
            startIcon={<InfoIcon />}
            fullWidth
            onClick={() => setShowInstructions(!showInstructions)}
          >
            {showInstructions ? "Hide Instructions" : "Show Instructions"}
          </UploadButton>
        </Box>

        {/* Instructions Section */}
        {showInstructions && (
          <Box mb={4}>
            <Typography
              variant="h6"
              sx={{
                mb: 2,
                color: "#2e7d32",
                fontSize: "1.6rem",
                fontStyle: "italic",
                textDecoration: "underline",
              }}
            >
              Follow these steps for a clear image:
            </Typography>
            {[
              "1. Choose a leaf with visible signs of disease.",
              "2. Place the leaf flat, and ensure it's in focus.",
              "3. Use natural light and avoid glare.",
              "4. Ensure the leaf fills the frame without distractions.",
            ].map((step, idx) => (
              <Typography
                key={idx}
                variant="body1"
                sx={{ mb: 1, color: "#388e3c", fontSize: "1.3rem" }}
              >
                {step}
              </Typography>
            ))}

            <Typography
              variant="body1"
              sx={{ mb: 3, color: "#388e3c", fontSize: "1.3rem" }}
            >
              📸 Upload your image in the following way to detect the disease.
            </Typography>
            <img
              src="/image/image1.png"
              alt="Example Leaf"
              style={{
                width: "100%",
                maxWidth: "300px",
                borderRadius: "12px",
                boxShadow: "0 4px 12px rgba(0, 0, 0, 0.2)",
              }}
            />
          </Box>
        )}
      </Container>
    </div>
  );
}

export default PlantDiseaseDetector;
