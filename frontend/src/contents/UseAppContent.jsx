import { Typography, Button, Grid, Card, CardContent, CardActions, Container, } from '@mui/material';
import { styled } from '@mui/system';

const StyledCard = styled(Card)(({ theme }) => ({
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
}));

const UseAppContent = () => {
    return (
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Grid container spacing={3}>
                {[1, 2, 3, 4].map((item) => (
                    <Grid item xs={12} sm={6} md={3} key={item}>
                        <StyledCard>
                            <CardContent>
                                <Typography variant="body2" color="text.secondary">
                                    Artificial intelligence application supports thyroid cancer classification according to The Bethesda
                                </Typography>
                            </CardContent>
                            <CardActions >
                                <Button size="small" variant="contained" color="primary">See detail</Button>
                            </CardActions>
                        </StyledCard>
                    </Grid>
                ))}
            </Grid>

            {/* Placeholder for additional content */}
            <div style={{ height: '200px', backgroundColor: '#8a2be2', marginTop: '20px' }}></div>
            <div style={{ height: '300px', backgroundColor: '#00bfff', marginTop: '20px' }}></div>
        </Container>
    );
}

export default UseAppContent;