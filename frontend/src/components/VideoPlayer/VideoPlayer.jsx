import React from 'react';
import ReactPlayer from 'react-player';

const VideoDemo = () => {
    return (
        <div style={{ height: '300px', backgroundColor: '#ffffff', marginTop: '20px'}}>
            <ReactPlayer
                // url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                url="https://www.youtube.com/watch?v=Bxdcmbe91Fg"
                width="100%"
                height="100%"
                controls
            />
        </div>
    );
};

export default VideoDemo;