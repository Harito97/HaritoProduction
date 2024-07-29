import { styled } from '@mui/material/styles';
import Paper from '@mui/material/Paper';
import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';

const StyledPaper = styled(Paper)({
    backgroundColor: '#ffffff',
    padding: '20px',
    borderRadius: '10px',
    marginTop: '20px',
    height: '400px',
    display: 'flex',
    flexDirection: 'column',
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

const ChatHistoryBox = styled(Paper)({
    flex: '1',
    backgroundColor: '#000000',
    color: 'white',
    padding: '10px',
    overflow: 'auto',
    marginBottom: '10px',
});

const ChatInputBox = styled(Paper)({
    backgroundColor: '#000000',
    display: 'flex',
    alignItems: 'center',
    padding: '10px',
});

const ChatBotBox = () => {
    return (
        <StyledPaper elevation={3}>
            <ChatHistoryBox elevation={0}>
                <Typography variant="body1">
                    [Assistant] Hello! I am a thyroid cancer classifier. How can I help you?
                </Typography>
            </ChatHistoryBox>
            <ChatInputBox elevation={0}>
                <TextField
                    variant="outlined"
                    placeholder="Type your message here..."
                    fullWidth
                    InputProps={{
                        style: {
                            color: 'white',
                        },
                    }}
                />
                <Button variant="contained" style={{ marginLeft: '10px' }}>
                    Send
                </Button>
            </ChatInputBox>
        </StyledPaper>
    );
};

export default ChatBotBox;