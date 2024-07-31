import React from "react";
import { Typography, Grid, Container, Box, Link } from '@mui/material';

const Footer = () => {
  return (
    <footer style={{ backgroundColor: '#333', color: '#fff', padding: '40px 0' }}>
      <Container maxWidth="lg">
        <Grid container spacing={4}>
          <Grid item xs={12} sm={4}>
            <Typography variant="h6" gutterBottom>Authors</Typography>
            <Box>
              <Typography>Pham Ngoc Hai</Typography>
              <Typography>Ass.Dr. Le Hong Phuong</Typography>
              <Typography>108 Military Central Hospital</Typography>
            </Box>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Typography variant="h6" gutterBottom>Other products</Typography>
            <Box display="flex" flexDirection="column" gap={1}>
              <Link href="#" color="inherit" underline="hover">Harito Thyroid cancer classifier</Link>
              <Link href="#" color="inherit" underline="hover">Harito Medical assistant</Link>
              <Link href="#" color="inherit" underline="hover">Harito Legal assistant</Link>
              <Link href="#" color="inherit" underline="hover">Harito Scientific research assistant</Link>
              <Link href="#" color="inherit" underline="hover">Harito Economic assistant</Link>
              <Link href="#" color="inherit" underline="hover">Other</Link>
            </Box>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Typography variant="h6" gutterBottom>Support us</Typography>
            <Box>
              <Typography>Bank: MB</Typography>
              <Typography>Account: 0325.982.517</Typography>
              <Typography>Name: PHAM NGOC HAI</Typography>
            </Box>
          </Grid>
        </Grid>
        <Box mt={4}>
          <Typography variant="body2" align="center">
            &copy; {new Date().getFullYear()} Harito. All rights reserved.
          </Typography>
        </Box>
      </Container>
    </footer>
  );
};

export default Footer;