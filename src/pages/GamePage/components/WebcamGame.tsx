import { observer } from 'mobx-react-lite';
import { useCallback, useRef, useState } from 'react';
import Webcam from 'react-webcam';
import styled from 'styled-components';
import axios from 'axios';
import './WebCam.css';
import { Point } from 'react-easy-crop/types';
import { API_URL } from '../../../constant/index';
import Cropper from 'react-easy-crop';
import { Direction } from '../../../model/Types';
import getCroppedImg from '../utils/cropImage';

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
    const [crop, setCrop] = useState<Point>({ x: -800, y: 300 });
    const webcamRef = useRef<any>(null);
    const [imgSrc, setImgSrc] = useState<any>(null);
    const onCropComplete = useCallback((croppedArea, croppedAreaPixels) => {
      console.log(croppedArea, croppedAreaPixels);
      // console.log(crop);
    }, []);

    const capture = useCallback(async () => {
      const imageSrc = webcamRef.current.getScreenshot({});
      setImgSrc(imageSrc);
      const imgCrop = await getCroppedImg(imageSrc, {
        width: 240,
        height: 240,
        x: 560,
        y: 0,
      });
      window.localStorage.setItem('img', imgCrop);

      const form = new FormData();
      form.append('imageFile', imgCrop);
      axios({
        method: 'post',
        url: API_URL,
        data: form,
        headers: { 'Content-Type': 'multipart/form-data' },
      })
        .then(function(response: any) {
          //handle success
          console.log(response);
        })
        .catch(function(response) {
          //handle error
          console.log(response);
        });
    }, [webcamRef, setImgSrc]);

    return (
      <Layout>
        <div className="my-video" style={{ position: 'relative' }}>
          <CaptureFrame />
          <Webcam
            audio={false}
            width={800}
            height={600}
            ref={webcamRef}
            screenshotFormat="image/png"
            mirrored={true}
            videoConstraints={videoConstraints}
          />
        </div>
        <button onClick={capture}>Capture Image</button>
        {imgSrc && (
          <Cropper
            image={imgSrc}
            crop={crop}
            cropSize={{ width: 240, height: 240 }}
            aspect={1}
            onCropChange={setCrop}
            onCropComplete={onCropComplete}
          />
        )}
      </Layout>
    );
  }
);

const CaptureFrame = styled.div`
  position: absolute;
  width: 240px;
  height: 240px;
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
