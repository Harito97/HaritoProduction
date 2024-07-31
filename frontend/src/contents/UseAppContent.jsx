import { useState } from 'react';
import { useEffect } from 'react';
import { Typography, Button, Grid, Card, CardContent, CardActions, Container, Modal, Box } from '@mui/material';
import { styled } from '@mui/system';
import { uploadImage } from '../services/api'; // Đảm bảo bạn đã định nghĩa uploadImage trong api.js
import ColumnChart from '../components/Chart/ColumnChart';
import ChatBotBox from '../components/ChatBotBox/ChatBotBox';
import VideoDemo from '../components/VideoPlayer/VideoPlayer';

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
    const [patchLevelHeatmap, setPatchLevelHeatmap] = useState(null);
    const [imageLevelHeatmap, setImageLevelHeatmap] = useState(null);
    const [yoloDetectImage, setYoloDetectImage] = useState(null);
    const [modalOpen, setModalOpen] = useState(false);
    const [modalImage, setModalImage] = useState(null);
    const [outputModel3, setOutputModel3] = useState(null);
    useEffect(() => {
        console.log("Updated outputModel3:", outputModel3);
    }, [outputModel3]);

    const handleUploadImage = async (event) => {
        const file = event.target.files[0];
        setUploadedImage(file);

        // Upload image and get results
        try {
            const formData = new FormData();
            formData.append('image', file);

            const response = await uploadImage(formData);
            console.log(response.data); // Log API response for debugging

            // Update state with response data
            setPatchLevelHeatmap(`data:image/png;base64,${response.data.patch_level_heatmap}`);
            setImageLevelHeatmap(`data:image/png;base64,${response.data.image_level_heatmap}`);
            setYoloDetectImage(`data:image/png;base64,${response.data.yolo_detect_image}`);

            // Process output_model3
            const processedOutputModel3 = response.data.output_model3[0].map((probability, index) => ({
                label: `Label ${index + 1}`,
                value: probability,
            }));
            setOutputModel3(processedOutputModel3); // Save processed output_model3
        } catch (error) {
            console.error("Error uploading image:", error);
        }
    };

    const handleOpenModal = (image) => {
        setModalImage(image);
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
                            {uploadedImage && (
                                <StyledImage
                                    src={URL.createObjectURL(uploadedImage)}
                                    alt="Uploaded Image"
                                    onClick={() => handleOpenModal(URL.createObjectURL(uploadedImage))}
                                />
                            )}
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
                            </Button>
                        </CardActions>
                    </StyledCard>
                </Grid>

                {/* Patch level */}
                <Grid item xs={12} sm={6} md={3} key="patch-level">
                    <StyledCard>
                        <CardContent>
                            {patchLevelHeatmap && (
                                <StyledImage
                                    src={patchLevelHeatmap}
                                    alt="Patch Level"
                                    onClick={() => handleOpenModal(patchLevelHeatmap)}
                                />
                            )}
                            {yoloDetectImage && (
                                <StyledImage
                                    src={yoloDetectImage}
                                    alt="YOLO Detection"
                                    onClick={() => handleOpenModal(yoloDetectImage)}
                                />
                            )}
                            <Typography variant="body2" color="text.secondary">
                                Patch level
                            </Typography>
                        </CardContent>
                    </StyledCard>
                </Grid>

                {/* Image level */}
                <Grid item xs={12} sm={6} md={3} key="image-level">
                    <StyledCard>
                        <CardContent>
                            {imageLevelHeatmap && (
                                <StyledImage
                                    src={imageLevelHeatmap}
                                    alt="Image Level Heatmap"
                                    onClick={() => handleOpenModal(imageLevelHeatmap)}
                                />
                            )}
                            {yoloDetectImage && (
                                <StyledImage
                                    src={yoloDetectImage}
                                    alt="YOLO Detection"
                                    onClick={() => handleOpenModal(yoloDetectImage)}
                                />
                            )}
                            <Typography variant="body2" color="text.secondary">
                                Image level
                            </Typography>
                        </CardContent>
                    </StyledCard>
                </Grid>
                
                <Grid item xs={12} sm={6} md={3} key="output-model">
                    <StyledCard>
                        <CardContent>
                            {Array.isArray(outputModel3) && outputModel3.length > 0 ? (
                                <ColumnChart data={outputModel3} />
                            ) : (
                                <Typography variant="body2" color="text.secondary">
                                    No data available
                                </Typography>
                            )}
                            <Typography variant="body2" color="text.secondary">
                                Output Model 3
                            </Typography>
                        </CardContent>
                    </StyledCard>
                </Grid>

            </Grid>

            {/* a chat bot box */}
            <ChatBotBox />
            {/* a area to show the video demo */}
            <VideoDemo />

            {/* Modal for displaying images */}
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
                    {modalImage && (
                        <StyledModalImage
                            src={modalImage}
                            alt="Uploaded Image"
                        />
                    )}
                </Box>
            </Modal>
        </Container>
    );
};

export default UseAppContent;


