// import React from "react";
// import { Typography, Grid, Container } from '@mui/material';

// const Footer = () => {
//   return (
//     <footer style={{ backgroundColor: '#f5f5f5', padding: '20px', marginTop: '20px' }}>
//       <Container maxWidth="lg">
//         <Grid container spacing={3}>
//           <Grid item xs={12} sm={4}>
//             <Typography variant="h6">Authors</Typography>
//             <Typography>Pham Ngoc Hai</Typography>
//             <Typography>Ass.Dr. Le Hong Phuong</Typography>
//             <Typography>108 Military Central Hospital</Typography>
//           </Grid>
//           <Grid item xs={12} sm={4}>
//             <Typography variant="h6">Other products</Typography>
//             <Typography>Harito Thyroid cancer classifier</Typography>
//             <Typography>Harito Medical assistant</Typography>
//             <Typography>Harito Legal assistant</Typography>
//             <Typography>Harito Scientific research assistant</Typography>
//             <Typography>Harito Economic assistant</Typography>
//             <Typography>Other</Typography>
//           </Grid>
//           <Grid item xs={12} sm={4}>
//             <Typography variant="h6">Support us</Typography>
//             <Typography>Bank: MB</Typography>
//             <Typography>Account: 0325.662.517</Typography>
//             <Typography>Name: PHAM NGOC HAI</Typography>
//           </Grid>
//         </Grid>
//       </Container>
//     </footer>
//   );
// };

// export default Footer;


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
              <Typography>Account: 0325.662.517</Typography>
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