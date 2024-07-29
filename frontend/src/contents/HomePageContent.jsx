import React from 'react';
import { Container, Typography, Button, Paper, Box, Grid } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const HomePageContent = () => {
    const navigate = useNavigate();

    const handleClick = () => {
        navigate('/use_app');
    };

    return (
        <Container maxWidth="md" style={{ marginTop: '2rem', marginBottom: '2rem' }}>
            <Paper
                elevation={3}
                sx={{
                    mt: 4,
                    mb: 4,
                    p: 4,
                    backgroundImage: `url(${process.env.PUBLIC_URL}/upload_img.jpg)`,
                    backgroundSize: 'cover',
                    backgroundPosition: 'center',
                    color: 'white',
                    position: 'relative',
                    '&:before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        backgroundColor: 'rgba(0, 0, 0, 0)',
                        zIndex: -1,
                    },
                }}
            >
                <Typography variant="h5" align="center" gutterBottom color={"#000000"}>
                    Artificial intelligence application supports thyroid cancer classification according to The Bethesda
                </Typography>
                <Box display="flex" justifyContent="center" mt={2}>
                    <Button variant="contained" color="secondary" onClick={handleClick} style={{ color: 'white' }}>
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