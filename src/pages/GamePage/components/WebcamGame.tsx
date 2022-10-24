import { observer } from 'mobx-react-lite';
import { useCallback, useRef } from 'react';
import Webcam from 'react-webcam';
import styled from 'styled-components';

import { Direction } from '../../../model/Types';

const videoConstraints = {
  width: 1280,
  height: 900,
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
      <div className="my-video">
        <Webcam
          audio={false}
          height={600}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width={800}
          mirrored={true}
          videoConstraints={videoConstraints}
        />
      </div>
    );
  }
);

const Layout = styled.div`
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
`;
export default WebcamGame;
