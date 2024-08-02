import React, { useState } from 'react';
import { styled } from '@mui/material/styles';
import Paper from '@mui/material/Paper';
// import Typography from '@mui/material/Typography';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import CircularProgress from '@mui/material/CircularProgress';
import ReactMarkdown from 'react-markdown';
import { chatResponse } from '../../services/api';

// Styled components
const StyledPaper = styled(Paper)({
    backgroundColor: '#ffffff',
    padding: '20px',
    borderRadius: '10px',
    marginTop: '20px',
    height: '500px',
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
    overflowY: 'auto',
    marginBottom: '10px',
    display: 'flex',
    flexDirection: 'column',
    '&::-webkit-scrollbar': {
        width: '8px',
    },
    '&::-webkit-scrollbar-track': {
        backgroundColor: '#333333',
    },
    '&::-webkit-scrollbar-thumb': {
        backgroundColor: '#666666',
        borderRadius: '4px',
    },
});

const ChatInputBox = styled(Paper)({
    backgroundColor: '#000000',
    display: 'flex',
    alignItems: 'center',
    padding: '10px',
    gap: '10px',
});

const StyledButton = styled(Button)({
    height: '56px',
    display: 'flex',
    alignItems: 'center',
});

const ChatBotBox = () => {
    const [messages, setMessages] = useState([
        { text: 'Hello! I am a thyroid cancer classifier. How can I help you?', sender: 'assistant' },
    ]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);

    const handleSend = async () => {
        if (input.trim()) {
            // Add user's message to the chat history
            setMessages([...messages, { text: input, sender: 'user' }]);

            // Set loading state to true
            setLoading(true);

            try {
                // Call the API using the imported function
                const response = await chatResponse({ message: input });

                // Add chatbot's response to the chat history
                setMessages([...messages, { text: input, sender: 'user' }, { text: response.data.response, sender: 'assistant' }]);
            } catch (error) {
                // Handle API error
                console.error('Error sending message:', error);
            } finally {
                // Clear input field and set loading to false
                setInput('');
                setLoading(false);
            }
        }
    };

    const handleNewChat = () => {
        setMessages([
            { text: 'Hello! I am a thyroid cancer classifier. How can I help you?', sender: 'assistant' },
        ]);
    };

    return (
        <StyledPaper elevation={3}>
            <ChatHistoryBox elevation={0}>
                {messages.map((message, index) => (
                    <div key={index} style={{
                        alignSelf: message.sender === 'user' ? 'flex-end' : 'flex-start',
                        backgroundColor: message.sender === 'user' ? '#4caf50' : '#1e88e5',
                        color: 'white',
                        borderRadius: '10px',
                        padding: '10px',
                        marginBottom: '5px',
                        maxWidth: '80%',
                        wordBreak: 'break-word',
                    }}>
                        <ReactMarkdown>{message.text}</ReactMarkdown>
                    </div>
                ))}
            </ChatHistoryBox>
            <ChatInputBox elevation={0}>
                <TextField
                    variant="outlined"
                    placeholder="Type your message here..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleSend()}
                    fullWidth
                    InputProps={{
                        style: {
                            color: 'white',
                        },
                        disabled: loading,
                    }}
                />
                <StyledButton
                    variant="contained"
                    color="primary"
                    onClick={handleSend}
                    disabled={loading}
                >
                    {loading ? <CircularProgress size={24} /> : 'Send'}
                </StyledButton>
                <StyledButton
                    variant="contained"
                    color="secondary"
                    onClick={handleNewChat}
                >
                    New Chat
                </StyledButton>
            </ChatInputBox>
        </StyledPaper>
    );
};

export default ChatBotBox;
