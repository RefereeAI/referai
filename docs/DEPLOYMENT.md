# Deployment Guide for *referai*

This document describes how the deployment process works.

## 1. Prerequisites

- Access to the project repository.
- Docker and Docker Compose installed.
- Environment variables configured (`.env`).

## 2. Clone the repository
```bash
git clone https://github.com/refereeai/referai
```

## 3. Set up environment variables
Copy the `.env.example` file to `.env` and edit it as needed.
    
- In the frontend: define environment variables needed for client to connect with the API. This should be the address from the backend.
- At the project root: you can define global or shared configuration variables.

Make sure not to commit these files to version control to protect sensitive information.

## 4. Build and start the services
```bash
docker-compose up --build -d
```

## 5. Check the status of the services
```bash
docker-compose ps
```

## 6. Access the application
Open your browser and go to `http://localhost:3000` (adjust the port if necessary).

## Production Deployment
Make sure to properly configure the production environment variables.

## Troubleshooting

To check the logs run:
```bash
docker-compose logs -f
```

## Updating

1. Run `git pull` to get the latest changes.
2. Execute:
    ```bash
    docker-compose up --build -d
    ```

---
For more detailed information, troubleshooting, or if you need to go back to the main documentation, please refer to the [README.md](../README.md).