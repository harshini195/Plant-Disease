import React from "react";
import { BrowserRouter, Routes, Route, Outlet } from "react-router-dom";
import "./App.css";
import PlantDiseaseDetector from "./components/PlantDiseaseDetector";
import Navbar from "./components/Navbar";

const Layout = () => {
  return (
    <div>
      <Navbar />
      <div className="container">
        <Outlet />
      </div>
    </div>
  );
};

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/">
          <Route path="" element={<PlantDiseaseDetector />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;