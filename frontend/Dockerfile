FROM node:18

WORKDIR /app

COPY package*.json ./

RUN npm install -g pnpm \
    && pnpm install

COPY . .

RUN pnpm run build

EXPOSE 3000

CMD ["pnpx", "serve", "-s", "build", "-l", "3000"]
