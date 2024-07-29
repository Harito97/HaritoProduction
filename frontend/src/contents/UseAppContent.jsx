import { useState } from 'react';
import { Typography, Button, Grid, Card, CardContent, CardActions, Container, Modal, Box } from '@mui/material';
import { styled } from '@mui/system';
import ChatBotBox from '../components/ChatBotBox/ChatBotBox';
import VideoDemo from '../components/VideoPlayer/VideoPlayer';
// import axios from 'axios';

const StyledCard = styled(Card)({
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
    alignItems: 'center',
    textAlign: 'center',
});

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

const StyledModalImage = styled('img')({
    maxWidth: '100%',
    maxHeight: '100%',
    objectFit: 'contain',
});


const UseAppContent = () => {
    const [uploadedImage, setUploadedImage] = useState(null);
    const [modalOpen, setModalOpen] = useState(false);

    const handleUploadImage = (event) => {
        setUploadedImage(event.target.files[0]);
    };

    const handleOpenModal = () => {
        setModalOpen(true);
    };

    const handleCloseModal = () => {
        setModalOpen(false);
    };

    return (
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Grid container spacing={3}>
                {/* Upload image */}
                <Grid item xs={12} sm={6} md={3} key="upload-image">
                    <StyledCard>
                        <CardContent>
                            <StyledImage
                                src={
                                    uploadedImage
                                        ? URL.createObjectURL(uploadedImage)
                                        : `${process.env.PUBLIC_URL}/upload_img.jpg`
                                }
                                alt="Uploaded Image"
                            />
                            <Typography variant="body2" color="text.secondary">
                                Upload image
                            </Typography>
                        </CardContent>
                        <CardActions sx={{ justifyContent: 'center' }}>
                            <Button
                                size="small"
                                variant="contained"
                                color="primary"
                                component="label"
                            >
                                Upload image
                                <input type="file" hidden onChange={handleUploadImage} />
                                {/* <input type="file" style={{ display: 'none' }} /> */}
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
                            <Button
                                size="small"
                                variant="contained"
                                color="primary"
                                onClick={handleOpenModal}
                            >
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
                            <Button
                                size="small"
                                variant="contained"
                                color="primary"
                                onClick={handleOpenModal}
                            >
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
                            <Button
                                size="small"
                                variant="contained"
                                color="primary"
                                onClick={handleOpenModal}
                            >
                                See detail
                            </Button>
                        </CardActions>
                    </StyledCard>
                </Grid>
            </Grid>

            {/* a chat bot box */}
            <ChatBotBox />
            {/* a area to show the video demo */}
            <VideoDemo />

            {/* Modal for displaying the uploaded image */}
            <Modal
                open={modalOpen}
                onClose={handleCloseModal}
                aria-labelledby="modal-modal-title"
                aria-describedby="modal-modal-description"
            >
                <Box
                    sx={{
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        transform: 'translate(-50%, -50%)',
                        bgcolor: 'background.paper',
                        border: '2px solid #000',
                        boxShadow: 24,
                        p: 4,
                        display: 'flex',
                        justifyContent: 'center',
                        alignItems: 'center',
                    }}
                >
                    {uploadedImage && (
                        <StyledModalImage
                            src={URL.createObjectURL(uploadedImage)}
                            alt="Uploaded Image"
                        />
                    )}
                </Box>
            </Modal>
        </Container>
    );
}

export default UseAppContent;
