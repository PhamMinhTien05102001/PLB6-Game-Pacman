import { observer } from 'mobx-react-lite';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import styled from 'styled-components';
import './WebCam.css';
import useWebSocket from 'react-use-websocket';
import { Point } from 'react-easy-crop/types';

import Cropper from 'react-easy-crop';
import { Direction } from '../../../model/Types';
import getCroppedImg from '../utils/cropImage';
import { useStore } from '../../../components/StoreContext';

const videoConstraints = {
  width: 800,
  height: 600,
  facingMode: 'user',
};

const captureConfig = {
  width: 300,
  height: 300,
  timeCapture: {
    low: 600,
    fast: 400,
  },
  acceptThreshold: 1,
  acceptPercent: 90,
};
const BackendUrl = 'ws://128.199.251.61:5001/';

type gestureCount = 'Attack' | 'Bottom' | 'Left' | 'Right' | 'Stop' | 'Top';

const WebcamGame = observer(
  ({
    triggerDirection,
    triggerGamePause,
    triggerChaos,
  }: {
    triggerDirection: (direction: Direction) => any;
    triggerGamePause: () => any;
    triggerChaos: () => any;
  }) => {
    const [crop, setCrop] = useState<Point>({ x: -800, y: -500 });
    const webcamRef = useRef<any>(null);
    const [timeReq, setTimeReq] = useState<any>();
    const [gesture, setGesture] = useState<string>();
    const store = useStore();

    const gestureCount = useMemo(() => {
      return {
        Attack: 0,
        Bottom: 0,
        Left: 0,
        Right: 0,
        Stop: 0,
        Top: 0,
      };
    }, []);

    const detectGesture = () => {
      // console.log(gestureCount);
      for (let i in gestureCount) {
        if (gestureCount[i as gestureCount] >= captureConfig.acceptThreshold) {
          resetGestureCount();
          setGesture(i);

          if (i === 'Bottom') triggerDirection('DOWN');
          if (i === 'Left') triggerDirection('LEFT');
          if (i === 'Right') triggerDirection('RIGHT');
          if (i === 'Top') triggerDirection('UP');
          if (i === 'Attack') triggerChaos();
          if (i === 'Stop') triggerGamePause();
        }
      }
    };
    const { sendMessage } = useWebSocket(BackendUrl, {
      onOpen: () => console.log('opened'),
      onMessage(event) {
        const timeRes = Date.now();
        console.log('New response message');
        console.log('Total time', timeRes - timeReq);

        const jsonEvent = JSON.parse(event.data.replaceAll("'", '"'));
        console.log('Time backend', jsonEvent['Time']);
        console.log(
          'Time reponse not include Backend',
          timeRes - timeReq - jsonEvent['Time'] * 1000
        );

        if (jsonEvent['Percent'] >= captureConfig.acceptPercent) {
          // console.log(response.data['Class Name'], 'Plus one');
          gestureCount[jsonEvent['Class Name'] as gestureCount] += 1;
        }
        detectGesture();
      },
    });

    const [imgSrc, setImgSrc] = useState<any>(null);

    useEffect(() => {
      const captureInterval = setInterval(
        () => {
          capture();
        },
        store.game.speed === 2
          ? captureConfig.timeCapture.fast
          : captureConfig.timeCapture.low
      );
      return () => clearInterval(captureInterval);
    }, [store.game.speed]);

    const resetGestureCount = () => {
      for (let i in gestureCount) {
        gestureCount[i as gestureCount] = 0;
      }
    };

    const onContinue = () => {
      store.game.gamePaused = false;
    };
    const capture = useCallback(async () => {
      const imageSrc = webcamRef.current.getScreenshot({});
      setImgSrc(imageSrc);
      const imgCrop = await getCroppedImg(imageSrc, {
        width: captureConfig.width,
        height: captureConfig.height,
        x: videoConstraints.width - captureConfig.width,
        y: videoConstraints.height - captureConfig.height,
      });
      const form = new FormData();
      form.append('imageFile', imgCrop);
      setTimeReq(Date.now());

      sendMessage(imgCrop);
    }, [webcamRef, setImgSrc]);

    return (
      <Layout>
        <div className="my-video" style={{ position: 'relative' }}>
          <CaptureFrame />
          {store.game.gamePaused ? (
            <button className="btn-video" onClick={onContinue}>
              Continue
            </button>
          ) : null}
          <Webcam
            audio={false}
            width={800}
            height={600}
            ref={webcamRef}
            screenshotFormat="image/png"
            className={store.game.gamePaused ? 'video-fallback' : ''}
            mirrored={true}
            videoConstraints={videoConstraints}
          />
        </div>
        <p style={{ color: 'white' }}>{gesture}</p>

        {imgSrc && (
          <Cropper
            image={imgSrc}
            crop={crop}
            cropSize={{
              width: captureConfig.width,
              height: captureConfig.height,
            }}
            aspect={1}
            onCropChange={setCrop}
          />
        )}
      </Layout>
    );
  }
);

const CaptureFrame = styled.div`
  position: absolute;
  width: ${captureConfig.width}px;
  height: ${captureConfig.height}px;
  border: 3px solid salmon;
  right: 0px;
  bottom: 3px;
  z-index: 20;
`;

const Layout = styled.div`
  display: flex;
  flex-direction: column;
  flex-wrap: wrap;
`;
export default WebcamGame;

// const onResults = (results: any) => {
//   // console.log(results);
//   setCountHandInBox(val => val + 1);

//   // if (countHandInBox === 5) {
//   //   setCountHandInBox(0);
//   //   console.log('Detect hand in box equal 5 and reset');
//   // }
// };

// useEffect(() => {
//   const hands = new Hands({
//     locateFile: file => {
//       return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`;
//     },
//   });
//   hands.setOptions({
//     maxNumHands: 1,
//     modelComplexity: 1,
//     minDetectionConfidence: 0.5,
//     minTrackingConfidence: 0.5,
//   });

//   hands.onResults(hand => {
//     // console.log(detectHandInBox(hand.multiHandLandmarks[0]));
//     if (detectHandInBox(hand.multiHandLandmarks[0])) {
//       onResults(hand);
//     }
//   });

//   if (
//     typeof webcamRef.current !== 'undefined' &&
//     webcamRef.current !== null
//   ) {
//     camera = new Camera(webcamRef.current.video, {
//       onFrame: async () => {
//         await hands.send({ image: webcamRef.current.video });
//       },
//       width: 800,
//       height: 600,
//     });
//     camera.start();
//   }
// }, []);
// const detectHandInBox = (finger: any) => {
//   if (!finger) return false;
//   for (let i = 0; i < finger.length; i++) {
//     if (finger[i].x > 0.3 || finger[i].y > 0.4) {
//       return false;
//     }
//   }
//   return true;
// };

// interface FingerCors {
//   x: number;
//   y: number;
//   z: number;
//   visibility: any;
// }
