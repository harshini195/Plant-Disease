import React from 'react';
import { useNavigate } from "react-router-dom";
import AppBar from "@mui/material/AppBar";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import GroupIcon from '@mui/icons-material/Group';
import HomeIcon from '@mui/icons-material/Home';
import PersonIcon from '@mui/icons-material/Person';
import Toolbar from "@mui/material/Toolbar";
import Typography from "@mui/material/Typography";


function Navbar() {
  const navigate = useNavigate();
  return (
    <Box sx={{ flexGrow: 1 }}>
      <AppBar position="static" color="info">
        <Toolbar>
          <Button color="inherit" onClick={() => navigate("/")} >
            <HomeIcon fontSize="large" />Expense Tracker<Typography
              variant="body1"
              component="div"
            >
            </Typography>
          </Button>
          <Box sx={{ flexGrow: 1 }} />
          <Button color="inherit" onClick={() => navigate("/login")} >
            <PersonIcon />Login
          </Button>
          <Button color="inherit" onClick={() => navigate("/register")} >
            <GroupIcon /> Register
          </Button>
        </Toolbar>
      </AppBar>
    </Box>
  );
}

export default Navbar;
