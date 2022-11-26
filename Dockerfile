FROM node:16-alpine3.15
WORKDIR /app
COPY . .
RUN yarn install
CMD ["npm", "start"]