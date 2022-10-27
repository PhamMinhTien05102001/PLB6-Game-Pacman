import { observer } from 'mobx-react-lite';
import { useCallback, useRef } from 'react';
import Webcam from 'react-webcam';
import styled from 'styled-components';

import { Direction } from '../../../model/Types';

const videoConstraints = {
  width: 800,
  height: 600,
  facingMode: 'user',
};

const WebcamGame = observer(
  ({
    triggerDirection,
  }: {
    triggerDirection: (direction: Direction) => any;
  }) => {
    const webcamRef = useRef<any>(null);

    const capture = useCallback(() => {
      webcamRef.current!.getScreenshot();
      triggerDirection('DOWN');
    }, [webcamRef]);
    return (
      <div className="my-video" style={{ position: 'relative' }}>
        <CaptureFrame />
        <Webcam
          audio={false}
          width={800}
          height={600}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          mirrored={true}
          videoConstraints={videoConstraints}
        />
      </div>
    );
  }
);
const CaptureFrame = styled.div`
  position: absolute;
  width: 400px;
  height: 300px;
  border: 5px solid salmon;
  right: 0;
  z-index: 20;
`;

const Layout = styled.div`
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
`;
export default WebcamGame;
