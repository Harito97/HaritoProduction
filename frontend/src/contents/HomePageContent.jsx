import React from 'react';
import { Container, Typography, Button, Paper, Box, Grid } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const HomePageContent = () => {
    const navigate = useNavigate();

    const handleClick = () => {
      navigate('/use_app');
    };

    return (
        <Container maxWidth="md">
            <Paper elevation={3} sx={{ mt: 4, mb: 4, p: 4, bgcolor: '#f5f5f5' }}>
                <Typography variant="h5" align="center" gutterBottom>
                    Artificial intelligence application supports thyroid cancer classification according to The Bethesda
                </Typography>
                <Box display="flex" justifyContent="center" mt={2}>
                    <Button variant="contained" color="secondary" onClick={handleClick}>
                        Try our app
                    </Button>
                </Box>
            </Paper>

            <Grid container spacing={3}>
                {[
                    { title: "Our impressive result", subtitle: "In the Thyroid Cancer Dataset of 108 Military Central Hospital" },
                    { title: "Current status of thyroid cancer", subtitle: "Globally" },
                    { title: "What better than other", subtitle: "Globally" },
                    { title: "How this AI work", subtitle: "Guide" },
                    { title: "Read paper", subtitle: "Link" },
                    { title: "Pricing to access", subtitle: "Money here" }
                ].map((item, index) => (
                    <Grid item xs={12} key={index}>
                        <Paper elevation={1} sx={{ p: 2 }}>
                            <Typography variant="h6">{item.title}</Typography>
                            <Typography variant="subtitle2" color="text.secondary">
                                {item.subtitle}
                            </Typography>
                        </Paper>
                    </Grid>
                ))}
            </Grid>
        </Container>
    );
};

export default HomePageContent;