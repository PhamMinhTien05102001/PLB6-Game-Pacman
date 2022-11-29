import { observer } from 'mobx-react-lite';
import React from 'react';
import styled from 'styled-components/macro';

export const Message = observer<{ className?: string; text: string }>(
  ({ className, text }) => {
    return <MessageStyled className={className}>{text}</MessageStyled>;
  }
);

const MessageStyled = styled.span`
  font-family: Joystix;
  font-size: 24px;
  color: yellow;
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  width: 220px;
  text-align: center;
`;
