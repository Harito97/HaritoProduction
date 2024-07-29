import React from "react";
import { Typography, Grid, Container } from '@mui/material';

const Footer = () => {
  return (
    <footer style={{ backgroundColor: '#f5f5f5', padding: '20px', marginTop: '20px' }}>
      <Container maxWidth="lg">
        <Grid container spacing={3}>
          <Grid item xs={12} sm={4}>
            <Typography variant="h5">Authors</Typography>
            <Typography>Pham Ngoc Hai</Typography>
            <Typography>Ass.Dr. Le Hong Phuong</Typography>
            <Typography>108 Military Central Hospital</Typography>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Typography variant="h5">Other products</Typography>
            <Typography>Harito Thyroid cancer classifier</Typography>
            <Typography>Harito Medical assistant</Typography>
            <Typography>Harito Legal assistant</Typography>
            <Typography>Harito Scientific research assistant</Typography>
            <Typography>Harito Economic assistant</Typography>
            <Typography>Other</Typography>
          </Grid>
          <Grid item xs={12} sm={4}>
            <Typography variant="h5">Support us</Typography>
            <Typography>Bank: MB</Typography>
            <Typography>Account: 0325.662.517</Typography>
            <Typography>Name: PHAM NGOC HAI</Typography>
          </Grid>
        </Grid>
      </Container>
    </footer>
  );
};

export default Footer;
