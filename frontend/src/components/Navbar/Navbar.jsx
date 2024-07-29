import React, { useState } from 'react';
import { AppBar, Toolbar, Typography, Button, IconButton, Menu, MenuItem, useMediaQuery, useTheme } from '@mui/material';
import MenuIcon from '@mui/icons-material/Menu';
import { useNavigate } from 'react-router-dom';

const Navbar = () => {
  const navigate = useNavigate();

  const handleClick = () => {
    navigate('/');
  };

  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [anchorEl, setAnchorEl] = useState(null);

  const handleMenu = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const menuItems = [
    { label: 'Intro app', href: '#' },
    { label: 'Solutions', href: '#' },
    { label: 'Read paper', href: '#' },
    { label: 'Statistics', href: '#' },
    { label: 'Pricing', href: '#' },
    { label: 'About us', href: '#' },
  ];

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }} onClick={handleClick}>
          Thyroid Cancer Classifier
        </Typography>
        {isMobile ? (
          <>
            <IconButton
              size="large"
              edge="start"
              color="inherit"
              aria-label="menu"
              onClick={handleMenu}
            >
              <MenuIcon />
            </IconButton>
            <Menu
              id="menu-appbar"
              anchorEl={anchorEl}
              anchorOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              open={Boolean(anchorEl)}
              onClose={handleClose}
            >
              {menuItems.map((item) => (
                <MenuItem key={item.label} onClick={handleClose}>
                  {item.label}
                </MenuItem>
              ))}
              <MenuItem onClick={handleClose}>Sign In</MenuItem>
              <MenuItem onClick={handleClose}>Register</MenuItem>
            </Menu>
          </>
        ) : (
          <>
            {menuItems.map((item) => (
              <Button key={item.label} color="inherit" href={item.href}>
                {item.label}
              </Button>
            ))}
            <Button color="inherit">Sign In</Button>
            <Button variant="contained" color="secondary">Register</Button>
          </>
        )}
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;