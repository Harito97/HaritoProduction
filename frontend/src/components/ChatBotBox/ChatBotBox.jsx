import { styled } from '@mui/material/styles';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';

const StyledPaper = styled(Paper)({
    backgroundColor: '#8a2be2',
    padding: '20px',
    borderRadius: '10px',
    marginTop: '20px',
    maxHeight: '200px',
    overflow: 'auto',
    '&::-webkit-scrollbar': {
        width: '10px',
    },
    '&::-webkit-scrollbar-track': {
        backgroundColor: '#8a2be2',
    },
    '&::-webkit-scrollbar-thumb': {
        backgroundColor: '#5a1d83',
        borderRadius: '5px',
    },
});

const ChatBotBox = () => {
    return (
        <StyledPaper elevation={3}>
            <Typography variant="body1" color="white">
                This is a chat bot box. You can add your chat messages here.
            </Typography>
        </StyledPaper>
    );
};

export default ChatBotBox;