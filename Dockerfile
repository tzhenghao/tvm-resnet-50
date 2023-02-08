# set base image
FROM public.ecr.aws/docker/library/python:3.10

# WORKDIR
ENV APP_HOME=/app
RUN mkdir -p $APP_HOME

# Set the working directory in the container
WORKDIR $APP_HOME

RUN apt-get update

# Copy the source files to the working directory
COPY . .

RUN pip install -r requirements.txt

# Run the container
CMD [ "echo", "done"]
