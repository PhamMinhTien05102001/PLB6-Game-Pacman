import { observer } from 'mobx-react-lite';
import { useCallback, useRef } from 'react';
import Webcam from 'react-webcam';
import styled from 'styled-components';
import { useStore } from '../../../components/StoreContext';
import { Direction } from '../../../model/Types';

const videoConstraints = {
  width: 1280,
  height: 720,
  facingMode: 'user',
};

const WebcamGame = observer(
  ({
    triggerDirection,
  }: {
    triggerDirection: (direction: Direction) => any;
  }) => {
    const webcamRef = useRef<any>(null);
    const store = useStore();
    const { game } = store;

    const pacMan = game.pacMan;
    const capture = useCallback(() => {
      webcamRef.current!.getScreenshot();
      triggerDirection('DOWN');
    }, [webcamRef]);
    return (
      <Layout>
        <Webcam
          audio={false}
          height={600}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          width={600}
          videoConstraints={videoConstraints}
        />
        <button
          onClick={capture}
          style={{ width: 'auto', background: 'blue', cursor: 'pointer' }}
        >
          Change direction
        </button>
      </Layout>
    );
  }
);

const Layout = styled.div`
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
`;
export default WebcamGame;
