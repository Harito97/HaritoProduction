import { Typography, Button, Grid, Card, CardContent, CardActions, Container } from '@mui/material';
import { styled } from '@mui/system';
import ChatBotBox from '../components/ChatBotBox/ChatBotBox';
import VideoDemo from '../components/VideoPlayer/VideoPlayer';

const StyledCard = styled(Card)(({ theme }) => ({
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    alignItems: 'center',
    textAlign: 'center',
}));

const StyledImage = styled('img')({
    width: '100px',
    height: '100px',
    border: '2px solid #4CAF50',
    borderRadius: '50%',
    padding: '4px',
    boxShadow: '0px 0px 10px 0px rgba(0,0,0,0.2)',
    transition: 'all 0.3s ease',
    '&:hover': {
        transform: 'scale(1.1)',
        boxShadow: '0px 0px 10px 0px rgba(0,0,0,0.4)',
    },
});

const UseAppContent = () => {
    return (
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Grid container spacing={3}>
                {/* Upload image */}
                <Grid item xs={12} sm={6} md={3} key="upload-image">
                    <StyledCard>
                        <CardContent>
                            <StyledImage src={`${process.env.PUBLIC_URL}/upload_img.jpg`} alt="Upload Image" />
                            <Typography variant="body2" color="text.secondary">
                                Upload image
                            </Typography>
                        </CardContent>
                        <CardActions sx={{ justifyContent: 'center' }}>
                            <Button size="small" variant="contained" color="primary">
                                Upload image
                            </Button>
                        </CardActions>
                    </StyledCard>
                </Grid>

                {/* Patch level */}
                <Grid item xs={12} sm={6} md={3} key="patch-level">
                    <StyledCard>
                        <CardContent>
                            <StyledImage src={`${process.env.PUBLIC_URL}/upload_img.jpg`} alt="Patch Level" />
                            <Typography variant="body2" color="text.secondary">
                                Patch level
                            </Typography>
                        </CardContent>
                        <CardActions sx={{ justifyContent: 'center' }}>
                            <Button size="small" variant="contained" color="primary">
                                See detail
                            </Button>
                        </CardActions>
                    </StyledCard>
                </Grid>

                {/* Plot */}
                <Grid item xs={12} sm={6} md={3} key="plot">
                    <StyledCard>
                        <CardContent>
                            <StyledImage src={`${process.env.PUBLIC_URL}/upload_img.jpg`} alt="Plot" />
                            <Typography variant="body2" color="text.secondary">
                                Plot
                            </Typography>
                        </CardContent>
                        <CardActions sx={{ justifyContent: 'center' }}>
                            <Button size="small" variant="contained" color="primary">
                                See detail
                            </Button>
                        </CardActions>
                    </StyledCard>
                </Grid>

                {/* Image level */}
                <Grid item xs={12} sm={6} md={3} key="image-level">
                    <StyledCard>
                        <CardContent>
                            <StyledImage src={`${process.env.PUBLIC_URL}/upload_img.jpg`} alt="Image Level" />
                            <Typography variant="body2" color="text.secondary">
                                Image level
                            </Typography>
                        </CardContent>
                        <CardActions sx={{ justifyContent: 'center' }}>
                            <Button size="small" variant="contained" color="primary">
                                See detail
                            </Button>
                        </CardActions>
                    </StyledCard>
                </Grid>
            </Grid>

            {/* Placeholder for additional content */}
            <ChatBotBox />
            {/* a chat bot box */}
            <VideoDemo />
            {/* a area to show the video demo */}
        </Container>
    );
}

export default UseAppContent;