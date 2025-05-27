FROM node:18

WORKDIR /app

COPY package*.json ./

RUN npm install -g pnpm \
    && pnpm install

COPY . .

ARG VITE_API_URL
ENV VITE_API_URL=${VITE_API_URL}

RUN VITE_API_URL=$VITE_API_URL pnpm run build

EXPOSE 3000

CMD ["pnpx", "serve", "-s", "dist", "-l", "3000"]
