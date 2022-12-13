import React from 'react';
import { render } from '@testing-library/react';
import { TestApp } from '../../test-util/TestApp';

describe('MazePage', () => {
  it('renders', () => {
    const { getByTestId } = render(<TestApp route="/" />);
    expect(getByTestId('GamePage')).toBeInTheDocument();
  });
});
